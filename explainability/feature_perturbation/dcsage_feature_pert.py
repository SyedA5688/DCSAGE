import os
import time
import json
import multiprocessing as mp

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from dataloader.feature_perturbation_dataloader import CovidUnperturbedDataset, CovidContainmentPerturbedDataset
from models.dcsage import DynamicAdjSAGE


def get_rolling_window_preds(args, SAVE_PATH, unperturbed_dataloader, min_containment_dataloader, max_containment_dataloader, model_idx):
    """
    Goal of function:
    For one model, return (num_windows, 3, 30, 10) - windows, 3 (unperturbed, min 
    containment, and max containment), 30 days of recursive predictions, 10 nodes.
    """

    print("Process", os.getpid(), ", Model", model_idx)
    ############
    # Load model
    ############
    if args["model_architecture"] == "DCSAGE":
        model = DynamicAdjSAGE(node_features=args['num_node_features'], emb_dim=args['embedding_dim'], window_size=args["window"], output=1, training=True, lstm_type=args["lstm_type"], name="DASAGE")
    else:
        raise NotImplementedError("Model architecture not implemeted.")

    checkpoint = torch.load(os.path.join(args["training_path"], args["training_run"], "model_" + str(model_idx) + ".pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    ######################################################################
    # Get 3 sets of recursive predictions for model on all rolling windows
    ######################################################################
    rolling_window_predictions_list = []

    with torch.no_grad():
        for roll_win_idx in range(len(unperturbed_dataloader.dataset) - args["recursive_pred_len"] - args["window"]):
            all_recursive_preds = []  # Will hold the three types of recursive preds for this rolling window

            # Unperturbed recursive predictions on rolling window
            unpert_recursive_preds = torch.zeros(args["recursive_pred_len"] + args["window"], 10)
            unpert_recursive_preds[0:args["window"],:] = torch.from_numpy(unperturbed_dataloader.dataset[roll_win_idx][0][:,:,1])

            for batch_window_node_feat, batch_window_edge_idx, batch_window_edge_attr, _ in unperturbed_dataloader:

                for count, window_idx in enumerate(range(roll_win_idx, roll_win_idx + args["recursive_pred_len"])):
                    window_node_feat = batch_window_node_feat[window_idx]
                    window_edge_idx = batch_window_edge_idx[window_idx]
                    window_edge_attr = batch_window_edge_attr[window_idx]
                    
                    if count > 0:
                        window_node_feat[:,:,1] = unpert_recursive_preds[count:count + args["window"]]

                    h_1, c_1, h_2, c_2 = None, None, None, None
                    for day_idx in range(len(window_node_feat)):
                        day_node_feat = window_node_feat[day_idx]
                        day_edge_idx = window_edge_idx[day_idx]
                        day_edge_attr = window_edge_attr[day_idx]

                        cutoff_idx = day_edge_idx[0].tolist().index(-1)
                        day_edge_idx = day_edge_idx[:, :cutoff_idx]
                        day_edge_attr = day_edge_attr[:cutoff_idx, :]
                        
                        day_graph = Data(x=day_node_feat, edge_index=day_edge_idx, edge_attr=day_edge_attr)
                        y_hat, h_1, c_1, h_2, c_2 = model(day_graph, h_1, c_1, h_2, c_2, day_idx)
                    unpert_recursive_preds[count + args["window"], :] = y_hat[:,0]
            unpert_recursive_preds = unpert_recursive_preds[args["window"]:,:].numpy()  # [30, 10]
            all_recursive_preds.append(unpert_recursive_preds)

        
            # Minimum containment recursive predictions on rolling window
            min_containment_recursive_preds = torch.zeros(args["recursive_pred_len"] + args["window"], 10)
            min_containment_recursive_preds[0:args["window"],:] = torch.from_numpy(min_containment_dataloader.dataset[roll_win_idx][0][:,:,1])

            for batch_window_node_feat, batch_window_edge_idx, batch_window_edge_attr, _ in min_containment_dataloader:

                for count, window_idx in enumerate(range(roll_win_idx, roll_win_idx + args["recursive_pred_len"])):
                    window_node_feat = batch_window_node_feat[window_idx]
                    window_edge_idx = batch_window_edge_idx[window_idx]
                    window_edge_attr = batch_window_edge_attr[window_idx]
                    
                    if count > 0:
                        window_node_feat[:,:,1] = min_containment_recursive_preds[count:count + args["window"]]

                    h_1, c_1, h_2, c_2 = None, None, None, None
                    for day_idx in range(len(window_node_feat)):
                        day_node_feat = window_node_feat[day_idx]
                        day_edge_idx = window_edge_idx[day_idx]
                        day_edge_attr = window_edge_attr[day_idx]

                        cutoff_idx = day_edge_idx[0].tolist().index(-1)
                        day_edge_idx = day_edge_idx[:, :cutoff_idx]
                        day_edge_attr = day_edge_attr[:cutoff_idx, :]
                        
                        day_graph = Data(x=day_node_feat, edge_index=day_edge_idx, edge_attr=day_edge_attr)
                        y_hat, h_1, c_1, h_2, c_2 = model(day_graph, h_1, c_1, h_2, c_2, day_idx)
                    min_containment_recursive_preds[count + args["window"], :] = y_hat[:,0]
            min_containment_recursive_preds = min_containment_recursive_preds[args["window"]:,:].numpy()
            all_recursive_preds.append(min_containment_recursive_preds)


            # Maximum containment recursive predictions on rolling window
            max_containment_recursive_preds = torch.zeros(args["recursive_pred_len"] + args["window"], 10)
            max_containment_recursive_preds[0:args["window"],:] = torch.from_numpy(max_containment_dataloader.dataset[roll_win_idx][0][:,:,1])

            for batch_window_node_feat, batch_window_edge_idx, batch_window_edge_attr, _ in max_containment_dataloader:

                for count, window_idx in enumerate(range(roll_win_idx, roll_win_idx + args["recursive_pred_len"])):
                    window_node_feat = batch_window_node_feat[window_idx]
                    window_edge_idx = batch_window_edge_idx[window_idx]
                    window_edge_attr = batch_window_edge_attr[window_idx]
                    
                    if count > 0:
                        window_node_feat[:,:,1] = max_containment_recursive_preds[count:count + args["window"]]

                    h_1, c_1, h_2, c_2 = None, None, None, None
                    for day_idx in range(len(window_node_feat)):
                        day_node_feat = window_node_feat[day_idx]
                        day_edge_idx = window_edge_idx[day_idx]
                        day_edge_attr = window_edge_attr[day_idx]

                        cutoff_idx = day_edge_idx[0].tolist().index(-1)
                        day_edge_idx = day_edge_idx[:, :cutoff_idx]
                        day_edge_attr = day_edge_attr[:cutoff_idx, :]
                        
                        day_graph = Data(x=day_node_feat, edge_index=day_edge_idx, edge_attr=day_edge_attr)
                        y_hat, h_1, c_1, h_2, c_2 = model(day_graph, h_1, c_1, h_2, c_2, day_idx)
                    max_containment_recursive_preds[count + args["window"], :] = y_hat[:,0]
            max_containment_recursive_preds = max_containment_recursive_preds[args["window"]:,:].numpy()
            all_recursive_preds.append(max_containment_recursive_preds)

            
            # Now aggregate 3 types of recursive pred (unpert, min containment, max containment) back into one array
            all_recursive_preds = np.array(all_recursive_preds)  # [3, 30, 10]
            rolling_window_predictions_list.append(all_recursive_preds)
    rolling_window_predictions_list = np.array(rolling_window_predictions_list)  # [num_windows, 3, 30, 10]

    return (model_idx, rolling_window_predictions_list)



model_process_results = []
def collect_results(result):
    global model_process_results
    model_process_results.append(result)


def main():
    """
    This function is the main driver of the program. It will create 3 dataloaders: one unperturbed, one with 
    max containment index value, and one with minimum containment index value
    """
    print("Parent process", os.getpid())

    unperturbed_dataset = CovidUnperturbedDataset(
        dataset_npz_path=args['dataset_npz_path'],
        window_size=args['window'], 
        data_split=args["dataset_split"], 
        avg_graph_structure=args["avg_graph_structure"])
    unperturbed_dataloader = DataLoader(unperturbed_dataset, batch_size=args['batch_size'], shuffle=False)

    min_containment_dataset = CovidContainmentPerturbedDataset(
        dataset_npz_path=args["dataset_npz_path"], 
        window_size=args["window"], 
        data_split=args["dataset_split"], 
        containment_new_val=0.1, 
        avg_graph_structure=args["avg_graph_structure"])
    min_containment_dataloader = DataLoader(min_containment_dataset, batch_size=args['batch_size'], shuffle=False)

    max_containment_dataset = CovidContainmentPerturbedDataset(
        dataset_npz_path=args["dataset_npz_path"], 
        window_size=args["window"], 
        data_split=args["dataset_split"], 
        containment_new_val=np.log10(100), 
        avg_graph_structure=args["avg_graph_structure"])
    max_containment_dataloader = DataLoader(max_containment_dataset, batch_size=args['batch_size'], shuffle=False)
    
    total_start_time = time.time()

    ##########################################################
    # Multiprocessing, each model becomes one separate process
    ##########################################################
    num_cpus = mp.cpu_count() - 2
    with mp.Pool(processes=num_cpus) as pool:
        for model_idx in range(args["num_models"]):
            pool.apply_async(
                get_rolling_window_preds, 
                args=(args, SAVE_PATH, unperturbed_dataloader, min_containment_dataloader, max_containment_dataloader, model_idx), 
                callback=collect_results)
        pool.close()
        pool.join()
    # get_rolling_window_preds(args, SAVE_PATH, unperturbed_dataloader, min_containment_dataloader, max_containment_dataloader, 0)

    # Callback has collected results from child processes, sort now by model index in tuple
    # Sorting by model index in tuple. 
    # Model_process_results is (100, tuple of (model_idx, rolling_window_predictions_list))
    # Each rolling_window_predictions_list is (num_windows, 3, 30, 10)
    model_process_results.sort(key=lambda x: x[0])  
    roll_win_feat_pert_predictions = []
    
    for roll_win_idx in range(len(unperturbed_dataloader.dataset) - args["recursive_pred_len"] - args["window"]):
        model_sublist = []  # Make (100, 3, 30, 10) for this rolling window
        for model_idx in range(args["num_models"]):
            model_sublist.append(model_process_results[model_idx][1][roll_win_idx])  # Appending [3,30,10]
        roll_win_feat_pert_predictions.append(model_sublist)
    
    os.mkdir(os.path.join(SAVE_PATH, "prediction_saves"))
    np.save(os.path.join(SAVE_PATH, "prediction_saves", "roll_win_feat_pert_preds.npy"), np.array(roll_win_feat_pert_predictions))

    print("\nTotal multiple-model node perturbation analysis took {:.3f} minutes".format((time.time() - total_start_time) / 60.))


if __name__ == "__main__":
    mp.set_start_method('spawn')
    with open("./explainability/feature_perturbation/feature_pert_config.json", "r") as f:
        args = json.load(f)

    if not os.path.exists(args["save_dir"]):
        os.mkdir(args["save_dir"])
    
    SAVE_PATH = os.path.join(args["save_dir"], args["training_run"])
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
        os.mkdir(os.path.join(SAVE_PATH, "python_file_saves"))
    
    os.system("cp feature_pert_config.json {}".format(os.path.join(SAVE_PATH, "python_file_saves")))
    os.system("cp dcsage_feature_pert.py {}".format(os.path.join(SAVE_PATH, "python_file_saves")))
    os.system("cp covid_feat_perturb_dataset.py {}".format(os.path.join(SAVE_PATH, "python_file_saves")))
    os.system("cp dcsage.py {}".format(os.path.join(SAVE_PATH, "python_file_saves")))
    os.system("cp weight_sage.py {}".format(os.path.join(SAVE_PATH, "python_file_saves")))

    main()


