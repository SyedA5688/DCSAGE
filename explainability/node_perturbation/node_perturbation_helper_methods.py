import os

import torch
import numpy as np
from torch_geometric.data import Data
from random import seed
from utils.node_perturbation_utils import *


chosen_seed = 0
seed(chosen_seed)
np.random.seed(chosen_seed)
torch.manual_seed(chosen_seed)


def perturbed_recursive_prediction_helper(model, perturbed_dataloaders, args, SAVE_PATH, model_idx=None, overwrite_window_idx=0, save_plot=True):
    """
    This function is a helper method for getting predictions from a model across all 
    perturbed dataloaders (10 continents can be perturbed). The function will obtain 
    30 days of recursive prediction starting at the specified window in the dataset
    (overwrite_window_idx), and will also obtain 30 days of regular prediction (not 
    recursively feeding predictions back to model) from the model. Regular and recursive 
    predictions will be accumulated in lists and turned into Pandas DataFrames, which 
    will be merged with other model's predictions in the main node perturbation script.

    Arguments:
    - model: model to get predictions from
    - perturbed_dataloaders: list of dataloaders, one for each continent which is perturbed
    - args: arguments dictionary
    - SAVE_PATH: path where predictions/figures should be saved
    - model_idx: index of model currently being trained, since node perturbation is 
            always run on multiple models
    - overwrite_window_idx: Index of window in dataset which should be fed to model 
            to initially start recursive prediction. This parameter allows this function
            to be reused for getting model predictions with different initial windows.
    - save_plot: whether or not to save plots for current model and initial window.
    """
    perturb_df_nested_lists = []

    with torch.no_grad():
        for idx, dataloader in enumerate(perturbed_dataloaders):
            # ============ Recursive Testing Loop ============
            extended_window_ncases = torch.zeros(args["recursive_pred_len"] + args["window"], 10)  # [44, 10]
            assert dataloader.dataset[overwrite_window_idx][0].shape[2] == 1, "Recursive prediction code made for 1 node feat currently"
            extended_window_ncases[0:args["window"],:] = torch.from_numpy(dataloader.dataset[overwrite_window_idx][0][:,:,0])

            for batch_window_node_feat, batch_window_edge_idx, batch_window_edge_attr, batch_window_labels in dataloader:

                for count, window_idx in enumerate(range(overwrite_window_idx, overwrite_window_idx + args["recursive_pred_len"])):
                    window_node_feat = batch_window_node_feat[window_idx]
                    window_edge_idx = batch_window_edge_idx[window_idx]
                    window_edge_attr = batch_window_edge_attr[window_idx]
                    window_labels = batch_window_labels[window_idx]
                    
                    if count > 0:
                        window_node_feat[:,:,0] = extended_window_ncases[count:count + args["window"]]

                    h_1, c_1, h_2, c_2 = None, None, None, None
                
                    for day_idx in range(len(window_node_feat)):
                        day_node_feat = window_node_feat[day_idx]
                        day_edge_idx = window_edge_idx[day_idx]
                        day_edge_attr = window_edge_attr[day_idx]

                        cutoff_idx = day_edge_idx[0].tolist().index(-1)
                        day_edge_idx = day_edge_idx[:, :cutoff_idx]
                        day_edge_attr = day_edge_attr[:cutoff_idx, :]
                        
                        # Forward through model
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
                        
                    extended_window_ncases[count + args["window"], :] = y_hat[:,0]
            
            extended_window_ncases = extended_window_ncases[args["window"]:,:].numpy()

            # ============ Regular Testing Loop ============
            predictions = []
            labels = []

            for batch_window_node_feat, batch_window_edge_idx, batch_window_edge_attr, batch_window_labels in dataloader:
                
                for window_idx in range(overwrite_window_idx, overwrite_window_idx + args["recursive_pred_len"]):
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
                        
                        # Forward through model
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
                    
                    predictions.append(y_hat.detach().unsqueeze(0).numpy())
                    labels.append(window_labels.detach().unsqueeze(0).numpy())
            
            predictions = np.concatenate(predictions, axis=0)
            labels = np.concatenate(labels, axis=0)

            assert extended_window_ncases.shape[0] == predictions.shape[0] == labels.shape[0], "Inconsistent sizes in predictions, recursive predictions, and ground truth labels"

            # ============ Plotting figures and Organizing Predictions into DataFrames ============
            # Plotting function returns a list of 10 dataframes, one for each continent, 
            # containing predictionss, labels, and recursive preds.
            if save_plot:
                trend_path = os.path.join(SAVE_PATH, "perturbed_trends")
                if not os.path.exists(trend_path):
                    os.mkdir(trend_path)
                pred_df_list = plot_and_return_perturbed_pred_df(predictions, labels, extended_window_ncases, 
                    perturbed_country_idx=idx, 
                    SAVE_PATH=trend_path, 
                    model_idx=model_idx)
            else:
                pred_df_list = get_pred_df(predictions, labels, extended_window_ncases)
            perturb_df_nested_lists.append(pred_df_list)
    return perturb_df_nested_lists


def unperturbed_recursive_prediction_helper(model, regular_dataloader, args, SAVE_PATH, model_idx=None, overwrite_window_idx=0, save_plot=True):
    """
    This function is a helper method for getting predictions from a model on the regular
    unperturbed dataloader. The function will obtain 30 days of recursive prediction 
    starting at the specified window in the dataset (overwrite_window_idx), and will also 
    obtain 30 days of regular prediction (not recursively feeding predictions back to 
    model) from the model. Regular and recursive predictions will be accumulated in lists 
    and turned into Pandas DataFrames, which will be merged with other model's predictions 
    in the main node perturbation script.

    Arguments:
    - model: model to get predictions from
    - regular_dataloaders: one regular dataloader with no perturbations performed
    - args: arguments dictionary
    - SAVE_PATH: path where predictions/figures should be saved
    - model_idx: index of model currently being trained, since node perturbation is 
            always run on multiple models
    - overwrite_window_idx: Index of window in dataset which should be fed to model 
            to initially start recursive prediction. This parameter allows this function
            to be reused for getting model predictions with different initial windows.
    - save_plot: whether or not to save plots for current model and initial window.
    """
    with torch.no_grad():
        # ============ Recursive Testing Loop ============
        assert regular_dataloader.dataset[overwrite_window_idx][0].shape[2] == 1, "Recursive prediction code made for 1 node feat currently"
        extended_window_ncases = torch.zeros(args["recursive_pred_len"] + args["window"], 10)
        extended_window_ncases[0:args["window"],:] = torch.from_numpy(regular_dataloader.dataset[overwrite_window_idx][0][:,:,0])

        for batch_window_node_feat, batch_window_edge_idx, batch_window_edge_attr, batch_window_labels in regular_dataloader:
            for count, window_idx in enumerate(range(overwrite_window_idx, overwrite_window_idx + args["recursive_pred_len"])):
                window_node_feat = batch_window_node_feat[window_idx]
                window_edge_idx = batch_window_edge_idx[window_idx]
                window_edge_attr = batch_window_edge_attr[window_idx]
                window_labels = batch_window_labels[window_idx]
                
                if count > 0:
                    assert args["num_node_features"] == 1, "Code optimized for 1 node feature."
                    window_node_feat[:,:,0] = extended_window_ncases[count:count + args["window"]]

                h_1, c_1, h_2, c_2 = None, None, None, None
                
                for day_idx in range(len(window_node_feat)):
                    day_node_feat = window_node_feat[day_idx]
                    day_edge_idx = window_edge_idx[day_idx]
                    day_edge_attr = window_edge_attr[day_idx]

                    cutoff_idx = day_edge_idx[0].tolist().index(-1)
                    day_edge_idx = day_edge_idx[:, :cutoff_idx]
                    day_edge_attr = day_edge_attr[:cutoff_idx, :]
                    
                    # Forward through model
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
                    
                extended_window_ncases[count + args["window"], :] = y_hat[:,0]
        
        extended_window_ncases = extended_window_ncases[args["window"]:,:].numpy()

        # ============ Regular Testing Loop ============
        predictions = []
        labels = []

        for batch_window_node_feat, batch_window_edge_idx, batch_window_edge_attr, batch_window_labels in regular_dataloader:
            
            for window_idx in range(overwrite_window_idx, overwrite_window_idx + args["recursive_pred_len"]):
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
                
                # Save prediction and label on each sliding window for evaluation later
                predictions.append(y_hat.detach().unsqueeze(0).numpy())
                labels.append(window_labels.detach().unsqueeze(0).numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        labels = np.concatenate(labels, axis=0)

        assert extended_window_ncases.shape[0] == predictions.shape[0] == labels.shape[0], "Inconsistent sizes in predictions, recursive predictions, and ground truth labels"

        # ============ Plotting figures and Organizing Predictions into DataFrames ============
        # Plotting function returns a list of 10 dataframes, one for each continent, 
        # containing predictionss, labels, and recursive preds.
        if save_plot:
            trend_path = os.path.join(SAVE_PATH, "unperturbed_trends")
            if not os.path.exists(trend_path):
                os.mkdir(trend_path)
            pred_df_list = plot_and_return_unperturbed_pred_df(predictions, labels, extended_window_ncases, 
                SAVE_PATH=trend_path, 
                model_idx=model_idx)
        else:
            pred_df_list = get_pred_df(predictions, labels, extended_window_ncases)
    
    return pred_df_list

