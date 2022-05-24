import os
import json
import time
import torch
import datetime
import multiprocessing as mp

from torch_geometric.data import Data
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from util import *
from test import test
from recursive_prediction import recursive_test
from covid_10country_dataset import Covid10CountriesDataset
from networks.dcsage import DynamicAdjSAGE
from networks.dcsage_gru import DCSAGE_GRU
from networks.dcsage_v2 import DCSAGE_v2
from networks.dcgat import DCGAT
from networks.dcgcn import DCGCN
from networks.dcgin import DCGIN
from networks.dcsage_temporal_attn import DCSAGE_Temporal_Attn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# def log_string(str1):
#     # Helper function to print string to stdout and also record it to training log text file
#     print(str1)
#     logger.write(str1 + "\n")
#     logger.flush()


def validation(args, SAVE_PATH, model, val_loader, loss_func, epoch_idx, lowest_validation_loss, model_idx):
    with torch.no_grad():
        model.eval()
        
        ################################
        # Evaluate on validation dataset
        ################################
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
                val_predictions.append(y_hat)
                val_labels.append(window_labels)

        ############################
        # Display validation results
        ############################
        val_avg_cost = val_total_cost / len(val_loader.dataset)
        if val_avg_cost < lowest_validation_loss:
            improved, improved_str = True, "(improved)"
        else:
            improved, improved_str = False, ""
        
        # log_string('Validation Loss: {:.5f} {}'.format(float(val_avg_cost), improved_str) + "\n")

        ####################################
        # Save model if there is improvement
        ####################################
        if improved:
            torch.save({
                'epoch': epoch_idx,
                'model_state_dict': model.state_dict(),
                'validation_loss': val_avg_cost
            }, os.path.join(SAVE_PATH, "model_" + str(model_idx) + ".pth"))
        
            return val_avg_cost, val_avg_cost, improved
        else:
            return val_avg_cost, lowest_validation_loss, improved


def train(args, SAVE_PATH, VISUALS_PATH, train_loader, val_loader, model, model_idx):
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

    # Get optimizer
    optimizer = None
    if args['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args["lr_scheduler_patience"], threshold=0.1, threshold_mode='rel')
    elif args['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args['learning_rate'], momentum=0.9)
        # Scheduler threshold is 0.2, so it is looking for validation loss to go below 0.8 * best_val_loss_recorded
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args["lr_scheduler_patience"], threshold=0.1, threshold_mode='rel')
    else:
        raise RuntimeError("Need to implement other optimizers besides Adam")
    
    model_visuals_path = os.path.join(VISUALS_PATH, "model_{}".format(model_idx))
    if not os.path.exists(model_visuals_path):
        os.mkdir(model_visuals_path)

    ###############
    # Training loop
    ###############
    lowest_validation_loss = 10000000
    lowest_train_loss = 10000000
    train_losses = []
    val_losses = []

    for epoch in range(args['epochs']):
        # log_string("Epoch " + str(epoch) + " starting...")
        train_total_cost = 0
        model.train()

        for batch_window_node_feat, batch_window_edge_idx, batch_window_edge_attr, batch_window_labels in train_loader:
            """
            batch_window_node_feat: torch.Tensor, shape [b, 14, 10, 3]
            batch_window_edge_idx: torch.Tensor, shape [b, 14, 2, 100]
            batch_window_edge_attr: torch.Tensor, shape [b, 14, 100, 1]
            batch_window_labels: torch.Tensor, shape [b, 10]
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

                total_batch_cost += loss_func(y_hat, window_labels)
                train_total_cost += loss_func(y_hat, window_labels)
            
            avg_batch_cost = total_batch_cost / batch_len
            avg_batch_cost.backward()
            if epoch % args["visual_freq"] == 0:
                model.plot_grad_flow(model_visuals_path, epoch)
                model.visualize_gradients("Train", model_visuals_path, epoch)
                # model.visualize_spatial_embeddings(args, train_loader, "Train", model_visuals_path, epoch)
                # model.visualize_activations(args, train_loader, "Train", model_visuals_path, epoch)
            optimizer.step()

        train_avg_cost = train_total_cost / len(train_loader.dataset)
        scheduler.step(train_avg_cost)
        train_losses.append(float(train_avg_cost))

        # if train_avg_cost < lowest_train_loss:
        #     train_improved_str = "(improved)"
        #     lowest_train_loss = train_avg_cost
        # else:
        #     train_improved_str = ""

        # log_string('Epoch: {:03d}, Learning rate: {}, Train Loss: {:.5f} {}'.format(epoch, optimizer.param_groups[0]['lr'], float(train_avg_cost), train_improved_str))

        validation_avg_cost, lowest_validation_loss, val_improved = validation(args, SAVE_PATH, model, val_loader, loss_func, epoch, lowest_validation_loss, model_idx)
        val_losses.append(float(validation_avg_cost))                
    
    plot_loss_curves(train_losses, val_losses, len(train_losses), os.path.join(SAVE_PATH, "loss_curves"), model_idx)


def multiprocessing_train_handler(args, SAVE_PATH, VISUALS_PATH, model_idx):
    """
    This function parallelizes the training of one model. One process will call this handler function,
    and that process will train 1 DCSAGE model.

    Args:
        - args:         arguments dictionary from covid10country_config.json
        - SAVE_PATH:    save directory for all models
        - model_idx:    index of model being trained by this process
    """
    print("Process", os.getpid(), ", Model", model_idx)

    # log_string("Loading dataset " + args['dataset_npz_path'])
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

    # log_string("Length of train dataset: " + str(len(train_dataset)))
    # log_string("Length of validation dataset: " + str(len(val_dataset)))
    # log_string("Length of test unsmooth dataset: " + str(len(test_dataset_unsmooth)) + "\n")

    train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False)
    test_unsmooth_dataloader = DataLoader(test_dataset_unsmooth, batch_size=args['batch_size'], shuffle=False)

    # log_string("Experiment description:\n" + args['experiment_desc'] + "\n")
    # log_string("Training configuration:")
    # for key in args:
    #     if key != "experiment_desc":
            # log_string(key + ": " + str(args[key]))
    # log_string("\n\n")

    # Define model
    if args["model_architecture"] == "DCSAGE":
        model = DynamicAdjSAGE(node_features=args['num_node_features'], emb_dim=args['embedding_dim'], window_size=args["window"], output=1, training=True, lstm_type=args["lstm_type"], name="DASAGE")
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
    elif args["model_architecture"] == "DCSAGE_v2":
        model = DCSAGE_v2(node_features=args['num_node_features'], emb_dim=args['embedding_dim'], window_size=args["window"], output=1, training=True)
    else:
        raise NotImplementedError("Model architecture not implemeted.")
    
    # log_string("---" * 30 + "\nBeginning training for model " + str(model_idx) + ":\n")
    # log_string("Model architecture:\n" + str(model) + "\n")
    # start_time = time.time()

    train(args, SAVE_PATH, VISUALS_PATH, train_dataloader, val_dataloader, model, model_idx)
    # log_string("Model training time was {:.3f} minutes.\n\n\n".format((time.time() - start_time) / 60.))
    
    model_path = os.path.join(SAVE_PATH, "model_" + str(model_idx) + ".pth")
    recursive_test_path = os.path.join(SAVE_PATH, "recursive_preds", "model_" + str(model_idx))
    test_unsmooth_path = os.path.join(SAVE_PATH, "test_unsmooth", "model_" + str(model_idx))
    os.mkdir(test_unsmooth_path)

    test(
        model_path, 
        args, 
        test_unsmooth_dataloader, 
        test_unsmooth_path, 
        split_name="Test_Unsmooth")

    recursive_test(
        saved_preds_dir=test_unsmooth_path,
        args=args,
        test_dataloader=test_unsmooth_dataloader,
        test_dataset=test_dataset_unsmooth,
        model_path=model_path,
        save_path=recursive_test_path)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    print("Parent process", os.getpid())
    with open("./covid10countries_config.json", "r") as f:
        args = json.load(f)
    args["num_models"] = 30  # 100

    assert args["model_architecture"] in ["DCSAGE", "DCSAGE_v2", "DCGAT", "DCSAGE_Temporal_Attn", "DCGCN", "DCGIN", "DCSAGE_GRU"]
    if not os.path.exists(args["save_dir"]):
        os.mkdir(args["save_dir"])
    
    date = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
    SAVE_PATH = os.path.join(args["save_dir"], date)
    VISUALS_PATH = os.path.join(SAVE_PATH, "visuals")
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
        os.mkdir(os.path.join(SAVE_PATH, "python_file_saves"))

    os.system("cp train_multiple_models.py {}".format(os.path.join(SAVE_PATH, "python_file_saves")))
    os.system("cp test.py {}".format(os.path.join(SAVE_PATH, "python_file_saves")))
    os.system("cp recursive_prediction.py {}".format(os.path.join(SAVE_PATH, "python_file_saves")))
    os.system("cp networks/dcsage.py {}".format(os.path.join(SAVE_PATH, "python_file_saves")))
    os.system("cp networks/weight_sage.py {}".format(os.path.join(SAVE_PATH, "python_file_saves")))
    os.system("cp covid_10country_dataset.py {}".format(os.path.join(SAVE_PATH, "python_file_saves")))
    os.system("cp covid10countries_config.json {}".format(os.path.join(SAVE_PATH, "python_file_saves")))

    # Multiple model training setup
    os.mkdir(os.path.join(SAVE_PATH, "visuals"))
    os.mkdir(os.path.join(SAVE_PATH, "test_unsmooth"))
    os.mkdir(os.path.join(SAVE_PATH, "recursive_preds"))
    os.mkdir(os.path.join(SAVE_PATH, "loss_curves"))
    total_start_time = time.time()

    # Multiprocessing
    with mp.Pool(processes=mp.cpu_count() - 2) as pool:
        for model_idx in range(args["num_models"]):
            pool.apply_async(
                multiprocessing_train_handler, 
                args=(args, SAVE_PATH, VISUALS_PATH, model_idx))
        pool.close()
        pool.join()
    # multiprocessing_train_handler(args, SAVE_PATH, VISUALS_PATH, 0)  # For debugging

    print("\nTotal multiple-model training time was {:.3f} minutes".format((time.time() - total_start_time) / 60.))
    
    # log_string("\n\n\nTotal multiple-model training time was {:.3f} minutes".format((time.time() - total_start_time) / 60.))
    # logger = open(os.path.join(SAVE_PATH, "training_log.txt"), "w")
    # main()
    # logger.close()
