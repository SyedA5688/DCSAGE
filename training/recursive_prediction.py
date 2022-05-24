import os
import json
import numpy as np

import torch
from torch_geometric.data import Data

from util import *
from networks.dcsage import DynamicAdjSAGE
from networks.dcsage_gru import DCSAGE_GRU
from networks.dcsage_v2 import DCSAGE_v2
from networks.dcgat import DCGAT
from networks.dcgcn import DCGCN
from networks.dcgin import DCGIN
from networks.dcsage_temporal_attn import DCSAGE_Temporal_Attn


def recursive_test(saved_preds_dir, args, test_dataloader, test_dataset, model_path, save_path):
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

    with open(os.path.join(saved_preds_dir, "test_predictions_save.json"), "r") as f:
        saved_dict = json.load(f)
    test_preds, test_labels = saved_dict['test_predictions'], saved_dict['test_labels']

    test_preds_np = np.array([np.array(nested_list) for nested_list in test_preds])
    test_labels_np = np.array([np.array(nested_list) for nested_list in test_labels])


    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        extended_window_ncases = torch.zeros(len(test_dataset) + args["window"], 10)
        extended_window_ncases[0:args["window"],:] = torch.from_numpy(test_dataset[0][0][:,:,0])  # Give first 14 real days

        for batch_window_node_feat, batch_window_edge_idx, batch_window_edge_attr, batch_window_labels in test_dataloader:  # idx from 0 to 82

            for window_idx in range(len(batch_window_node_feat)):
                window_node_feat = batch_window_node_feat[window_idx]  # shape [14, 10, 3]
                window_edge_idx = batch_window_edge_idx[window_idx]  # shape [14, 2, 100]
                window_edge_attr = batch_window_edge_attr[window_idx]  # shape [14, 100, 1]
                window_labels = batch_window_labels[window_idx]  # shape [10]
                
                # Overwrite num_cases node feature with model predictions starting at 2nd window
                if window_idx > 0:
                    window_node_feat[:,:,0] = extended_window_ncases[window_idx:window_idx + args["window"]]

                h_1, c_1, h_2, c_2 = None, None, None, None
                for day_idx in range(len(window_node_feat)):
                    day_node_feat = window_node_feat[day_idx]  # shape [10, 3]
                    day_edge_idx = window_edge_idx[day_idx]  # shape [2, 100]
                    day_edge_attr = window_edge_attr[day_idx]  # shape [100, 1]

                    cutoff_idx = day_edge_idx[0].tolist().index(-1)
                    day_edge_idx = day_edge_idx[:, :cutoff_idx]
                    day_edge_attr = day_edge_attr[:cutoff_idx, :]
                    
                    # Construct graph object for a single day inside of the sliding window, pass through model
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
                    
                # Append predictions to end of extended_window_num_cases
                extended_window_ncases[window_idx + args["window"], :] = y_hat[:,0]
        
        # Remove first 14 days that were given from data. Matches length with labels and real predictions on test set, good for plotting
        extended_window_ncases = extended_window_ncases[args["window"]:,:]

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        plot_continent_pred_vs_ground_truth_vs_extended_feedback_pred_trend(test_preds_np, test_labels_np, extended_window_ncases, SAVE_PATH=save_path)
