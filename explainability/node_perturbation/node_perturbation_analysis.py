import os
import json
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from random import seed

from utils.node_perturbation_utils import *
from models.dcsage import DynamicAdjSAGE
from models.dcsage_gru import DCSAGE_GRU
from models.dcgat import DCGAT
from models.dcgcn import DCGCN
from models.dcgin import DCGIN
from models.dcsage_temporal_attn import DCSAGE_Temporal_Attn
from dataloader.node_perturbation_dataloader import Covid10CountriesUnperturbedDataset, Covid10CountriesPerturbedDataset


chosen_seed = 0
seed(chosen_seed)
np.random.seed(chosen_seed)
torch.manual_seed(chosen_seed)


def generate_perturbed_recursive_pred_dfs(model, perturbed_dataloaders, args, SAVE_PATH, model_idx=None, overwrite_window_idx=0, save_plot=True):
    """
    model_idx only applies if we are training multiple models. If passed, then plot saving will differentiate by model #.
    overwrite_window_idx is 0 for all single model analysis (meaning first show model first window of test dataset, predict
        recursive predictions after that). For rolling window multiple-model analysis, we need change which initial window
        we feed to model to start recursive predictions, hence overwrite_window_idx.
    """
    perturb_df_nested_lists = []

    with torch.no_grad():
        for idx, dataloader in enumerate(perturbed_dataloaders):
            ##############################
            # Run recursive testing loop #
            ##############################
            extended_window_ncases = torch.zeros(args["recursive_pred_len"] + args["window"], 10)  # [44, 10]
            assert dataloader.dataset[overwrite_window_idx][0].shape[2] == 1, "Recursive prediction code made for 1 node feat currently"
            extended_window_ncases[0:args["window"],:] = torch.from_numpy(dataloader.dataset[overwrite_window_idx][0][:,:,0])

            for batch_window_node_feat, batch_window_edge_idx, batch_window_edge_attr, batch_window_labels in dataloader:

                for count, window_idx in enumerate(range(overwrite_window_idx, overwrite_window_idx + args["recursive_pred_len"])):
                    window_node_feat = batch_window_node_feat[window_idx]  # shape [14, 10, 3]
                    window_edge_idx = batch_window_edge_idx[window_idx]  # shape [14, 2, 100]
                    window_edge_attr = batch_window_edge_attr[window_idx]  # shape [14, 100, 1]
                    window_labels = batch_window_labels[window_idx]  # shape [10]
                    
                    if count > 0:
                        window_node_feat[:,:,0] = extended_window_ncases[count:count + args["window"]]

                    h_1, c_1, h_2, c_2 = None, None, None, None
                
                    for day_idx in range(len(window_node_feat)):
                        day_node_feat = window_node_feat[day_idx]  # shape [10, 3]
                        day_edge_idx = window_edge_idx[day_idx]  # shape [2, 100]
                        day_edge_attr = window_edge_attr[day_idx]  # shape [100, 1]

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
                        
                    extended_window_ncases[count + args["window"], :] = y_hat[:,0]
            
            extended_window_ncases = extended_window_ncases[args["window"]:,:].numpy()

            ############################
            # Run regular testing loop #
            ############################
            predictions = []
            labels = []

            for batch_window_node_feat, batch_window_edge_idx, batch_window_edge_attr, batch_window_labels in dataloader:
                
                for window_idx in range(overwrite_window_idx, overwrite_window_idx + args["recursive_pred_len"]):
                    window_node_feat = batch_window_node_feat[window_idx]  # shape [14, 10, 3]
                    window_edge_idx = batch_window_edge_idx[window_idx]  # shape [14, 2, 100]
                    window_edge_attr = batch_window_edge_attr[window_idx]  # shape [14, 100, 1]
                    window_labels = batch_window_labels[window_idx]  # shape [10]

                    h_1, c_1, h_2, c_2 = None, None, None, None
                
                    for day_idx in range(len(window_node_feat)):
                        day_node_feat = window_node_feat[day_idx]  # shape [10, 3]
                        day_edge_idx = window_edge_idx[day_idx]  # shape [2, 100]
                        day_edge_attr = window_edge_attr[day_idx]  # shape [100, 1]

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
                    
                    predictions.append(y_hat.detach().unsqueeze(0).numpy())
                    labels.append(window_labels.detach().unsqueeze(0).numpy())
            
            predictions = np.concatenate(predictions, axis=0)
            labels = np.concatenate(labels, axis=0)

            assert extended_window_ncases.shape[0] == predictions.shape[0] == labels.shape[0], "Inconsistent sizes in predictions, recursive predictions, and ground truth labels"

            # Plotting function returns a list of 10 dataframes, one for each country containing preds, labels, and recursive preds
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


def generate_regular_recursive_pred_df(model, regular_dataloader, args, SAVE_PATH, model_idx=None, overwrite_window_idx=0, save_plot=True):
    """
    model_idx only applies if we are training multiple models. If passed, then plot saving will differentiate by model #.
    overwrite_window_idx is 0 for all single model analysis (meaning first show model first window of test dataset, predict
        recursive predictions after that). For rolling window multiple-model analysis, we need change which initial window
        we feed to model to start recursive predictions, hence overwrite_window_idx.
    """
    with torch.no_grad():
        ##############################
        # Run recursive testing loop #
        ##############################
        assert regular_dataloader.dataset[overwrite_window_idx][0].shape[2] == 1, "Recursive prediction code made for 1 node feat currently"
        extended_window_ncases = torch.zeros(args["recursive_pred_len"] + args["window"], 10)
        extended_window_ncases[0:args["window"],:] = torch.from_numpy(regular_dataloader.dataset[overwrite_window_idx][0][:,:,0])

        for batch_window_node_feat, batch_window_edge_idx, batch_window_edge_attr, batch_window_labels in regular_dataloader:
            for count, window_idx in enumerate(range(overwrite_window_idx, overwrite_window_idx + args["recursive_pred_len"])):
                window_node_feat = batch_window_node_feat[window_idx]  # shape [14, 10, 3]
                window_edge_idx = batch_window_edge_idx[window_idx]  # shape [14, 2, 100]
                window_edge_attr = batch_window_edge_attr[window_idx]  # shape [14, 100, 1]
                window_labels = batch_window_labels[window_idx]  # shape [10]
                
                if count > 0:
                    assert args["num_node_features"] == 1, "Code optimized for 1 node feature."
                    window_node_feat[:,:,0] = extended_window_ncases[count:count + args["window"]]

                h_1, c_1, h_2, c_2 = None, None, None, None
                
                for day_idx in range(len(window_node_feat)):
                    day_node_feat = window_node_feat[day_idx]  # shape [10, 3]
                    day_edge_idx = window_edge_idx[day_idx]  # shape [2, 100]
                    day_edge_attr = window_edge_attr[day_idx]  # shape [100, 1]

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
                    
                extended_window_ncases[count + args["window"], :] = y_hat[:,0]
        
        extended_window_ncases = extended_window_ncases[args["window"]:,:].numpy()

        ############################
        # Run regular testing loop #
        ############################
        predictions = []
        labels = []

        for batch_window_node_feat, batch_window_edge_idx, batch_window_edge_attr, batch_window_labels in regular_dataloader:
            
            for window_idx in range(overwrite_window_idx, overwrite_window_idx + args["recursive_pred_len"]):
                window_node_feat = batch_window_node_feat[window_idx]  # shape [14, 10, 3]
                window_edge_idx = batch_window_edge_idx[window_idx]  # shape [14, 2, 100]
                window_edge_attr = batch_window_edge_attr[window_idx]  # shape [14, 100, 1]
                window_labels = batch_window_labels[window_idx]  # shape [10]

                h_1, c_1, h_2, c_2 = None, None, None, None
                
                for day_idx in range(len(window_node_feat)):
                    day_node_feat = window_node_feat[day_idx]  # shape [10, 3]
                    day_edge_idx = window_edge_idx[day_idx]  # shape [2, 100]
                    day_edge_attr = window_edge_attr[day_idx]  # shape [100, 1]

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

        # Plotting function returns a list of 10 dataframes, one for each country containing preds, labels, and recursive preds
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


def node_perturbation_difference_heatmap(perturb_df_nested_lists, regular_df_nested_list, save_plot=True):
    """
    perturb_df_nested_lists: (10, 10, 30, 4) - 10 perturbed countries, 10 countries in graph, pd.DataFrame of shape (30, 4)
        - 30 days of prediction on test set (after first 14 days)
        - 4 columns are 'Regular Predictions', 'Ground Truth', 'Extended Recursive Predictions', and 'Day Index'
    
    regular_df_nested_list: (10, 30, 4)
    """
    aggreg_differences_lists = []

    for perturbed_country_idx in range(10):
        aggreg_differences = []
        for country_idx in range(10):
            if country_idx == perturbed_country_idx:
                aggreg_differences.append(np.nan)
            else:
                # Debug and check that this is 30 days long
                difference_list = perturb_df_nested_lists[perturbed_country_idx][country_idx]['Extended Recursive Predictions'] - regular_df_nested_list[country_idx]['Extended Recursive Predictions']
                aggreg_differences.append(difference_list.sum())

        aggreg_differences_lists.append(aggreg_differences)
    
    if save_plot:
        plot_difference_heatmap(aggreg_differences_lists, SAVE_PATH)
    return aggreg_differences_lists


def main():
    # Define 10 perturbed dataloaders
    perturbed_dataloaders = []
    for idx in range(10):
        dataset_unsmooth = Covid10CountriesPerturbedDataset(
            dataset_npz_path=args["dataset_npz_path"], 
            window_size=args["window"], 
            data_split=args["dataset_split"], 
            perturb_country_idx=idx, 
            avg_graph_structure=args["avg_graph_structure"])
        dataloader = DataLoader(dataset_unsmooth, batch_size=args['batch_size'], shuffle=False)
        perturbed_dataloaders.append(dataloader)
    
    # Define one regular unperturbed dataloaders
    unperturbed_dataset = Covid10CountriesUnperturbedDataset(
        dataset_npz_path=args['dataset_npz_path'],
        window_size=args['window'], 
        data_split=args["dataset_split"], 
        savg_graph_structure=args["avg_graph_structure"])
    unperturbed_dataloader = DataLoader(unperturbed_dataset, batch_size=args['batch_size'], shuffle=False)
    
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
    else:
        raise NotImplementedError("Model architecture not implemeted.")

    checkpoint = torch.load(os.path.join(args["training_path"], args["training_run"], "best_model.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    perturb_df_nested_lists = generate_perturbed_recursive_pred_dfs(model, perturbed_dataloaders, args, SAVE_PATH)
    regular_df_nested_list = generate_regular_recursive_pred_df(model, unperturbed_dataloader, args, SAVE_PATH)

    aggreg_differences_lists = node_perturbation_difference_heatmap(perturb_df_nested_lists, regular_df_nested_list)
    # node_perturbation_sensitivity_barchart(aggreg_differences_lists)

    plot_recursive_and_perturbed_recursive_per_country(perturb_df_nested_lists, regular_df_nested_list, SAVE_PATH)


if __name__ == "__main__":
    """
    Note: Only run node_perturbation_analysis.py directly if you want to do perturbation analysis on a single
    trained model. In most cases, you will instead want to run analysis on many models using the 
    multiple_model_pert_analysis.py script.
    """
    with open("/Users/syedrizvi/Desktop/Projects/GNN_Project/DCSAGE/Node-Perturbation/node_perturb_analysis_config.json", "r") as f:
        args = json.load(f)
    args["training_path"] = "/Users/syedrizvi/Desktop/Projects/GNN_Project/DCSAGE/Training-Code/training-runs/"

    if not os.path.exists(args["save_dir"]):
        os.mkdir(args["save_dir"])
    
    date = args["training_run"]
    SAVE_PATH = os.path.join(args["save_dir"], date)
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
        os.mkdir(os.path.join(SAVE_PATH, "python_file_saves"))
    
    os.system("cp node_perturb_analysis_config.json {}".format(SAVE_PATH))
    os.system("cp node_perturbation_analysis.py {}".format(os.path.join(SAVE_PATH, "python_file_saves")))
    os.system("cp covid_10country_perturb_dataset.py {}".format(os.path.join(SAVE_PATH, "python_file_saves")))

    main()
    