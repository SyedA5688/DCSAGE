import os
import json
import time
import torch
import datetime
import numpy as np

from utils.training_utils import *
from test import test
from recursive_prediction import recursive_test

from torch_geometric.data import Data
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataloader.training_dataloader import Covid10CountriesDataset
from models.dcsage import DynamicAdjSAGE
from models.dcsage_gru import DCSAGE_GRU
from models.dcgcn import DCGCN
from models.dcgin import DCGIN
from models.dcgat import DCGAT
from models.dcsage_temporal_attn import DCSAGE_Temporal_Attn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def log_string(str1):
    print(str1)
    logger.write(str1 + "\n")
    logger.flush()


def validation(model, val_loader, loss_func, epoch_idx, lowest_validation_loss):
    with torch.no_grad():
        model.eval()

        # ============ Validation loop ============
        val_total_cost = 0
        val_predictions = []
        val_labels = []

        for batch_window_node_feat, batch_window_edge_idx, batch_window_edge_attr, batch_window_labels in val_loader:
            
            for window_idx in range(len(batch_window_node_feat)):
                window_node_feat = batch_window_node_feat[window_idx]
                window_edge_idx = batch_window_edge_idx[window_idx]
                window_edge_attr = batch_window_edge_attr[window_idx]
                window_labels = batch_window_labels[window_idx]

                h_1, c_1, h_2, c_2 = None, None, None, None
                for day_idx in range(len(window_node_feat)):
                    day_node_feat = window_node_feat[day_idx]
                    day_edge_idx = window_edge_idx[day_idx]
                    day_edge_attr = window_edge_attr[day_idx]

                    cutoff_idx = day_edge_idx[0].tolist().index(-1)
                    day_edge_idx = day_edge_idx[:, :cutoff_idx]
                    day_edge_attr = day_edge_attr[:cutoff_idx, :]
                    
                    day_graph = Data(x=day_node_feat, edge_index=day_edge_idx, edge_attr=day_edge_attr)
                    if args["model_architecture"] in ["DCSAGE", "DCGCN", "DCGIN"]:
                        y_hat, h_1, c_1, h_2, c_2 = model(day_graph, h_1, c_1, h_2, c_2, day_idx)
                    elif args["model_architecture"] == "DCSAGE_GRU":
                        y_hat, h_1, h_2 = model(day_graph, h_1, h_2, day_idx)
                    elif args["model_architecture"] == "DCGAT":
                        y_hat, h_1, c_1, h_2, c_2, _ = model(day_graph, h_1, c_1, h_2, c_2, day_idx)
                    elif args["model_architecture"] == "DCSAGE_Temporal_Attn":
                        y_hat, h_1, c_1, h_2, c_2, _ = model(day_graph, h_1, c_1, h_2, c_2, day_idx)
                    else:
                        y_hat = model(day_graph, day_idx, target_seq=window_labels)

                val_total_cost += loss_func(y_hat, window_labels)
                # Save prediction and labels on each window
                val_predictions.append(y_hat)
                val_labels.append(window_labels)

        # ============ Log validation loss ============
        val_avg_cost = val_total_cost / len(val_loader.dataset)
        if val_avg_cost < lowest_validation_loss:
            improved, improved_str = True, "(improved)"
        else:
            improved, improved_str = False, ""
        
        log_string('Validation Loss: {:.5f} {}'.format(float(val_avg_cost), improved_str) + "\n")

        # ============ Save model on improvement ============
        if improved:
            # Save colored correlation plot
            save_colored_correlation_pred_vs_label_plot(val_predictions, val_labels, plot_title='Validation Correlation Plot, Epoch {}, Validation loss {:.5f}'.format(str(epoch_idx), float(val_avg_cost)), plot_save_name='val_epoch{}_corr_colored_plot'.format(epoch_idx), SAVE_PATH=os.path.join(SAVE_PATH, "visuals", "val_corr_plots"))

            # Save model, overwrite previous best checkpoint
            torch.save({
                'epoch': epoch_idx,
                'model_state_dict': model.state_dict(),
                'validation_loss': val_avg_cost
            }, os.path.join(SAVE_PATH, "best_model.pth"))
        
            return val_avg_cost, val_avg_cost, improved
        else:
            return val_avg_cost, lowest_validation_loss, improved


def train(train_loader, val_loader, model):
    # ============ Define loss function ============
    name_to_loss_func = {
        "mape_loss": mape_loss,
        "mse_loss": mse_loss,
        "msse_loss": msse_loss,
        "rmse_loss": rmse_loss,
        "mae_loss": mae_loss,
        "mase_loss": mase_loss,
        "mase1_loss": mase1_loss,
        "mase2_loss": mase2_loss,
        "anti_lag_loss": anti_lag_loss,
    }

    assert args['loss_function'] in name_to_loss_func, "Unknown loss function specified in configuration file"
    loss_func = name_to_loss_func[args['loss_function']]

    # ============ Log experiment information and model architecture ============
    log_string("Experiment description:\n" + args['experiment_desc'] + "\n")
    log_string("Model architecture:\n" + str(model) + "\n")

    log_string("Training configuration:")
    for key in args:
        if key != "experiment_desc":
            log_string(key + ": " + str(args[key]))
    log_string("\n\n")

    # ============ Define optimizer ============
    optimizer = None
    if args['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), args['learning_rate'])
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args["lr_scheduler_patience"], threshold=0.1, threshold_mode='rel')
    elif args['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args['learning_rate'], momentum=0.9)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args["lr_scheduler_patience"], threshold=0.1, threshold_mode='rel')
    else:
        raise RuntimeError("Need to implement other optimizers besides Adam")

    # ============ Training loop ============
    lowest_validation_loss = 10000000
    lowest_train_loss = 10000000

    for epoch in range(args['epochs']):
        train_total_cost = 0
        model.train()

        # If train_loader batch size > num dataset samples --> batch gradient descent
        for batch_window_node_feat, batch_window_edge_idx, batch_window_edge_attr, batch_window_labels in train_loader:
            """
            batch_window_node_feat: torch.Tensor, shape [b, window_len, n_nodes, n_features]
            batch_window_edge_idx: torch.Tensor, shape [b, window_len, 2, 100]
            batch_window_edge_attr: torch.Tensor, shape [b, window_len, 100, 1]
            batch_window_labels: torch.Tensor, shape [b, n_nodes]
            """
            batch_len = len(batch_window_node_feat)
            total_batch_cost = 0
            optimizer.zero_grad()
            
            for window_idx in range(len(batch_window_node_feat)):
                window_node_feat = batch_window_node_feat[window_idx]
                window_edge_idx = batch_window_edge_idx[window_idx]
                window_edge_attr = batch_window_edge_attr[window_idx]
                window_labels = batch_window_labels[window_idx]

                h_1, c_1, h_2, c_2 = None, None, None, None

                for day_idx in range(len(window_node_feat)):
                    day_node_feat = window_node_feat[day_idx]
                    day_edge_idx = window_edge_idx[day_idx]
                    day_edge_attr = window_edge_attr[day_idx]

                    cutoff_idx = day_edge_idx[0].tolist().index(-1)
                    day_edge_idx = day_edge_idx[:, :cutoff_idx]
                    day_edge_attr = day_edge_attr[:cutoff_idx, :]
                    
                    # Forward 1-day graph through model
                    day_graph = Data(x=day_node_feat, edge_index=day_edge_idx, edge_attr=day_edge_attr)
                    if args["model_architecture"] in ["DCSAGE", "DCGCN", "DCGIN"]:
                        y_hat, h_1, c_1, h_2, c_2 = model(day_graph, h_1, c_1, h_2, c_2, day_idx)
                    elif args["model_architecture"] == "DCSAGE_GRU":
                        y_hat, h_1, h_2 = model(day_graph, h_1, h_2, day_idx)
                    elif args["model_architecture"] == "DCGAT":
                        y_hat, h_1, c_1, h_2, c_2, _ = model(day_graph, h_1, c_1, h_2, c_2, day_idx)
                    elif args["model_architecture"] == "DCSAGE_Temporal_Attn":
                        y_hat, h_1, c_1, h_2, c_2, _ = model(day_graph, h_1, c_1, h_2, c_2, day_idx)
                    else:
                        y_hat = model(day_graph, day_idx, target_seq=window_labels)

                # Sum loss over each day in window
                total_batch_cost += loss_func(y_hat, window_labels)
                train_total_cost += loss_func(y_hat, window_labels)
            
            avg_batch_cost = total_batch_cost / batch_len
            avg_batch_cost.backward()
            if epoch % args["visual_freq"] == 0:
                model.plot_grad_flow(VISUALS_PATH, epoch)
                model.visualize_gradients("Train", VISUALS_PATH, epoch)
                # model.visualize_spatial_embeddings(args, train_loader, "Train", VISUALS_PATH, epoch)
                # model.visualize_activations(args, train_loader, "Train", VISUALS_PATH, epoch)
            optimizer.step()

        # ============ Calculate and log training loss ============
        train_avg_cost = train_total_cost / len(train_loader.dataset)
        scheduler.step(train_avg_cost)
        train_losses.append(float(train_avg_cost))

        if train_avg_cost < lowest_train_loss:
            train_improved_str = "(improved)"
            lowest_train_loss = train_avg_cost
        else:
            train_improved_str = ""

        log_string('Epoch: {:03d}, Learning rate: {}, Train Loss: {:.5f} {}'.format(epoch, optimizer.param_groups[0]['lr'], float(train_avg_cost), train_improved_str))

        # ============ Validation ============
        validation_avg_cost, lowest_validation_loss, val_improved = validation(model, val_loader, loss_func, epoch, lowest_validation_loss)
        val_losses.append(float(validation_avg_cost))
    
    # ============ Plot loss curves after training ends, save final model ============
    plot_loss_curves(train_losses, val_losses, len(train_losses), SAVE_PATH=SAVE_PATH)

    model.eval()
    torch.save({
        'epoch': args["epochs"],
        'model_state_dict': model.state_dict()
    }, os.path.join(SAVE_PATH, "last_model.pth"))


def main():
    # =================== Define datasets ===================
    log_string("Loading dataset " + args['dataset_npz_path'])

    train_dataset = Covid10CountriesDataset(
        dataset_npz_path=args['dataset_npz_path'],
        window_size=args['window'], 
        data_split="train", 
        avg_graph_structure=args["avg_graph_structure"])

    val_dataset = Covid10CountriesDataset(
        dataset_npz_path=args['dataset_npz_path'],
        window_size=args['window'], 
        data_split="validation", 
        avg_graph_structure=args["avg_graph_structure"])
    
    test_dataset_unsmooth = Covid10CountriesDataset(
        dataset_npz_path=args['dataset_npz_path'],
        window_size=args['window'], 
        data_split="test-unsmooth", 
        avg_graph_structure=args["avg_graph_structure"])

    log_string("Length of train dataset: " + str(len(train_dataset)))
    log_string("Length of validation dataset: " + str(len(val_dataset)))
    log_string("Length of test unsmooth dataset: " + str(len(test_dataset_unsmooth)) + "\n")

    train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False)
    test_unsmooth_dataloader = DataLoader(test_dataset_unsmooth, batch_size=args['batch_size'], shuffle=False)

    # ============= Define model =============
    if args["model_architecture"] == "DCSAGE":
        model = DynamicAdjSAGE(node_features=args['num_node_features'], emb_dim=args['embedding_dim'], window_size=args["window"], output=1, training=True, lstm_type=args["lstm_type"], name="DCSAGE")
    elif args["model_architecture"] == "DCSAGE_GRU":
        model = DCSAGE_GRU(node_features=args['num_node_features'], emb_dim=args['embedding_dim'], window_size=args["window"], output=1, training=True, name="DCSAGE_GRU")
    elif args["model_architecture"] == "DCGAT":
        model = DCGAT(node_features=args['num_node_features'], emb_dim=args['embedding_dim'], window_size=args["window"], output=1, name="DASAGE")
    elif args["model_architecture"] == "DCSAGE_Temporal_Attn":
        model = DCSAGE_Temporal_Attn(node_features=args['num_node_features'], emb_dim=args['embedding_dim'], window_size=args["window"], output=1, training=True, lstm_type=args["lstm_type"])
    elif args["model_architecture"] == "DCGCN":
        model = DCGCN(node_features=args['num_node_features'], emb_dim=args['embedding_dim'], window_size=args["window"], output=1, training=True, lstm_type=args["lstm_type"], name="DCGCN")
    elif args["model_architecture"] == "DCGIN":
        model = DCGIN(node_features=args['num_node_features'], emb_dim=args['embedding_dim'], window_size=args["window"], output=1, training=True, lstm_type=args["lstm_type"], name="DCGIN")
    else:
        raise NotImplementedError("Model architecture not implemeted.")

    # ================ Training loop ================
    start_time = time.time()
    try:
        train(train_dataloader, val_dataloader, model)
    except KeyboardInterrupt:
        # Plot train-val loss curves if training interrupted through keyboard
        min_len = min(len(train_losses), len(val_losses))
        plot_loss_curves(train_losses[:min_len], val_losses[:min_len], len(train_losses), SAVE_PATH=SAVE_PATH)
    
    log_string("Training took %s seconds" % (time.time() - start_time))
    
    # Only continue to run test on train/val/test if a best validation model was saved, otherwise exit.
    assert os.path.isfile(os.path.join(SAVE_PATH, "best_model.pth")), "No best model was saved, exiting without running test on any dataset split."
    
    # ============ Create directories for test evaluations ============
    os.mkdir(os.path.join(SAVE_PATH, "test_unsmooth"))  # Test on unsmoothened data
    os.mkdir(os.path.join(SAVE_PATH, "test_smooth"))  # Test on smoothened data
    os.mkdir(os.path.join(SAVE_PATH, "validation"))
    os.mkdir(os.path.join(SAVE_PATH, "training"))

    # Run test for unsmooth and smooth test set, training set, and validation set
    model_path = os.path.join(SAVE_PATH, "best_model.pth")
    test(model_path, args, test_unsmooth_dataloader, os.path.join(SAVE_PATH, "test_unsmooth"), split_name="Test_Unsmooth")
    test(model_path, args, val_dataloader, os.path.join(SAVE_PATH, "validation"), split_name="Validation")
    test(model_path, args, train_dataloader, os.path.join(SAVE_PATH, "training"), split_name="Training")

    # Run recursive prediction
    recursive_test(saved_preds_dir=os.path.join(SAVE_PATH, "test_unsmooth"),
        args=args,
        test_dataloader=test_unsmooth_dataloader,
        test_dataset=test_dataset_unsmooth,
        model_path=model_path,
        save_path=os.path.join(SAVE_PATH, "recursive_preds"))


if __name__ == "__main__":
    # ============ Read in training options from config file ============
    with open("./training/training_config.json", "r") as f:
        args = json.load(f)
    args["save_dir"] = "./training/training-runs"
    
    assert args["model_architecture"] in ["DCSAGE", "DCSAGE_v2", "DCGAT", "DCSAGE_Temporal_Attn", "DCGCN", "DCGIN", "DCSAGE_GRU"]
    
    if not os.path.exists(args["save_dir"]):
        os.mkdir(args["save_dir"])
    
    # ============ Create training experiment directories ============
    date = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
    SAVE_PATH = os.path.join(args["save_dir"], date)
    VISUALS_PATH = os.path.join(SAVE_PATH, "visuals")
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
        os.mkdir(os.path.join(SAVE_PATH, "visuals"))
        os.mkdir(os.path.join(SAVE_PATH, "visuals", "val_corr_plots"))
        os.mkdir(os.path.join(SAVE_PATH, "python_file_saves"))

    # ============ Copy files to experiment directory for record ============
    os.system("cp training/train.py {}".format(os.path.join(SAVE_PATH, "python_file_saves")))
    os.system("cp training/test.py {}".format(os.path.join(SAVE_PATH, "python_file_saves")))
    os.system("cp training/training_config.json {}".format(os.path.join(SAVE_PATH, "python_file_saves")))
    os.system("cp training/recursive_prediction.py {}".format(os.path.join(SAVE_PATH, "python_file_saves")))
    os.system("cp dataloader/training_dataloader.py {}".format(os.path.join(SAVE_PATH, "python_file_saves")))
    os.system("cp models/dcsage.py {}".format(os.path.join(SAVE_PATH, "python_file_saves")))
    os.system("cp models/weight_sage.py {}".format(os.path.join(SAVE_PATH, "python_file_saves")))

    train_losses = []
    val_losses = []

    logger = open(os.path.join(SAVE_PATH, "training_log.txt"), "w")
    main()
    logger.close()
