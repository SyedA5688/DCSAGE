import os
import json
import numpy as np

import torch
from torch_geometric.data import Data

from utils.training_utils import *
from models.dcsage import DynamicAdjSAGE
from models.dcsage_gru import DCSAGE_GRU
from models.dcgat import DCGAT
from models.dcgcn import DCGCN
from models.dcgin import DCGIN
from models.dcsage_temporal_attn import DCSAGE_Temporal_Attn


def recursive_test(saved_preds_dir, args, test_dataloader, test_dataset, model_path, save_path):
    """
    This function runs recursive prediction evaluation on the test dataset using the best
    model from training. This function should be run after regular test evaluation.

    Arguments:
    - saved_preds_dir: path to directory containing saved predictions on test set
    - args: arguments dictionary
    - test_dataloader: the test dataloader
    - test dataset: test dataset, needed separately for accessing some arrays
    - model_path: path to best trained model from training
    - save_path: path to directory where recursive prediction trends should be saved
    """
    # ============ Define model ============
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

    # ============ Load test set predictions saved earlier by test function ============
    with open(os.path.join(saved_preds_dir, "test_predictions_save.json"), "r") as f:
        saved_dict = json.load(f)
    test_preds, test_labels = saved_dict['test_predictions'], saved_dict['test_labels']

    test_preds_np = np.array([np.array(nested_list) for nested_list in test_preds])
    test_labels_np = np.array([np.array(nested_list) for nested_list in test_labels])

    # ============ Load best checkpoint from training ============
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        # ============ Recursive test loop ============
        extended_window_ncases = torch.zeros(len(test_dataset) + args["window"], 10)
        # Give first 7 days of data to model to start recursive prediction
        extended_window_ncases[0:args["window"],:] = torch.from_numpy(test_dataset[0][0][:,:,0])

        for batch_window_node_feat, batch_window_edge_idx, batch_window_edge_attr, batch_window_labels in test_dataloader:

            for window_idx in range(len(batch_window_node_feat)):
                window_node_feat = batch_window_node_feat[window_idx]
                window_edge_idx = batch_window_edge_idx[window_idx]
                window_edge_attr = batch_window_edge_attr[window_idx]
                window_labels = batch_window_labels[window_idx]
                
                # Overwrite num_cases feature with model predictions
                if window_idx > 0:
                    window_node_feat[:,:,0] = extended_window_ncases[window_idx:window_idx + args["window"]]

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
                    
                # Add model predictions to predictions array
                extended_window_ncases[window_idx + args["window"], :] = y_hat[:,0]
        
        # Remove first 7 days that were given from data, keep only recursive predictions
        extended_window_ncases = extended_window_ncases[args["window"]:,:]

        # ============ Plot recursive prediction trend ============
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        plot_continent_pred_vs_ground_truth_vs_extended_feedback_pred_trend(test_preds_np, test_labels_np, extended_window_ncases, SAVE_PATH=save_path)
