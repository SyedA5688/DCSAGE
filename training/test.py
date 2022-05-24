import os
import json
import torch

from util import *
from torch_geometric.data import Data

from networks.dcsage import DynamicAdjSAGE
from networks.dcsage_gru import DCSAGE_GRU
from networks.dcsage_v2 import DCSAGE_v2
from networks.dcgat import DCGAT
from networks.dcgcn import DCGCN
from networks.dcgin import DCGIN
from networks.dcsage_temporal_attn import DCSAGE_Temporal_Attn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test(model_path, args, data_loader, SAVE_PATH, split_name):
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

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():        
        ##########################
        # Evaluate on test dataset
        ##########################
        predictions = []
        labels = []
        window_alphas_attn_coeff1 = []
        window_alphas_attn_coeff2 = []

        for batch_window_node_feat, batch_window_edge_idx, batch_window_edge_attr, batch_window_labels in data_loader:
            
            for window_idx in range(len(batch_window_node_feat)):
                window_node_feat = batch_window_node_feat[window_idx]  # shape [14, 10, 3]
                window_edge_idx = batch_window_edge_idx[window_idx]  # shape [14, 2, 100]
                window_edge_attr = batch_window_edge_attr[window_idx]  # shape [14, 100, 1]
                window_labels = batch_window_labels[window_idx]  # shape [10]

                day_alphas_attn_coeff1 = []
                day_alphas_attn_coeff2 = []

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
                        y_hat, h_1, c_1, h_2, c_2, alphas_info = model(day_graph, h_1, c_1, h_2, c_2, day_idx)
                        (alphas_edge_index1, attention_weights1, alphas_edge_index2, attention_weights2) = alphas_info

                        matr = np.zeros((10, 10))
                        matr2 = np.zeros((10, 10))
                        for i in range(alphas_edge_index1.shape[1]):
                            matr[alphas_edge_index1[0,i], alphas_edge_index1[1,i]] = attention_weights1[i].mean()
                            matr2[alphas_edge_index2[0,i], alphas_edge_index2[1,i]] = attention_weights2[i].mean()
                        
                        for j in range(10):  # Zero out diagonal
                            matr[j,j] = 0
                            matr2[j,j] = 0

                        day_alphas_attn_coeff1.append(matr)
                        day_alphas_attn_coeff2.append(matr2)
                    elif args["model_architecture"] == "DCSAGE_Temporal_Attn":
                        y_hat, h_1, c_1, h_2, c_2, (h1_alphas, h2_alphas) = model(day_graph, h_1, c_1, h_2, c_2, day_idx)
                        # h1_alphas and h2_alphas are shape (seq_len, seq_len). Row is attn coefficients for 1 day in sequence against all other days.
                    else:
                        y_hat = model(day_graph, day_idx, target_seq=window_labels)
                
                if args["model_architecture"] == "DCGAT":
                    window_alphas_attn_coeff1.append(day_alphas_attn_coeff1)
                    window_alphas_attn_coeff2.append(day_alphas_attn_coeff2)
                elif args["model_architecture"] == "DCSAGE_Temporal_Attn":
                    window_alphas_attn_coeff1.append(np.array(h1_alphas))
                    window_alphas_attn_coeff2.append(np.array(h2_alphas))
                # Save prediction and label on each sliding window for evaluation later
                predictions.append(y_hat)
                labels.append(window_labels)
        
        if args["model_architecture"] == "DCGAT":
            window_alphas_attn_coeff1 = np.array(window_alphas_attn_coeff1)
            np.save(os.path.join(SAVE_PATH, "attn_coeff_1_matr.npy"), window_alphas_attn_coeff1)
            window_alphas_attn_coeff2 = np.array(window_alphas_attn_coeff2)
            np.save(os.path.join(SAVE_PATH, "attn_coeff_2_matr.npy"), window_alphas_attn_coeff2)

            # Save heatmap
            plot_spatial_attn_coeff_heatmap(window_alphas_attn_coeff1.mean(axis=0).mean(axis=0), title="GAT Layer 1 Mean Spatial Attention Coefficient Heatmap", save_path=os.path.join(SAVE_PATH, "gat_layer1_mean_attn_heatmap.png"))
            plot_spatial_attn_coeff_heatmap(window_alphas_attn_coeff1.sum(axis=0).sum(axis=0), title="GAT Layer 1 Summed Spatial Attention Coefficient Heatmap", save_path=os.path.join(SAVE_PATH, "gat_layer1_summed_attn_heatmap.png"))
            plot_spatial_attn_coeff_heatmap(window_alphas_attn_coeff2.mean(axis=0).mean(axis=0), title="GAT Layer 2 Mean Spatial Attention Coefficient Heatmap", save_path=os.path.join(SAVE_PATH, "gat_layer2_mean_attn_heatmap.png"))
            plot_spatial_attn_coeff_heatmap(window_alphas_attn_coeff2.sum(axis=0).sum(axis=0), title="GAT Layer 2 Summed Spatial Attention Coefficient Heatmap", save_path=os.path.join(SAVE_PATH, "gat_layer2_summed_attn_heatmap.png"))
        elif args["model_architecture"] == "DCSAGE_Temporal_Attn":
            window_alphas_attn_coeff1 = np.array(window_alphas_attn_coeff1)
            np.save(os.path.join(SAVE_PATH, "temporal_attn_coeff_1_matr.npy"), window_alphas_attn_coeff1)
            window_alphas_attn_coeff2 = np.array(window_alphas_attn_coeff2)
            np.save(os.path.join(SAVE_PATH, "temporal_attn_coeff_2_matr.npy"), window_alphas_attn_coeff2)

            # Save heatmap
            # plot_temporal_attn_coeff_heatmap(window_alphas_attn_coeff1.mean(axis=0), seq_len=args["window"], title="LSTM Cell 1 Mean Temporal Attention Coefficient Heatmap", save_path=os.path.join(SAVE_PATH, "lstm_cell1_mean_attn_heatmap.png"))
            # plot_temporal_attn_coeff_heatmap(window_alphas_attn_coeff1.sum(axis=0), seq_len=args["window"], title="LSTM Cell 1 Summed Temporal Attention Coefficient Heatmap", save_path=os.path.join(SAVE_PATH, "lstm_cell1_summed_attn_heatmap.png"))
            plot_temporal_attn_coeff_heatmap(window_alphas_attn_coeff2.mean(axis=0), seq_len=args["window"], title="LSTM Cell 2 Mean Temporal Attention Coefficient Heatmap", save_path=os.path.join(SAVE_PATH, "lstm_cell2_mean_attn_heatmap.png"))
            plot_temporal_attn_coeff_heatmap(window_alphas_attn_coeff2.sum(axis=0), seq_len=args["window"], title="LSTM Cell 2 Summed Temporal Attention Coefficient Heatmap", save_path=os.path.join(SAVE_PATH, "lstm_cell2_summed_attn_heatmap.png"))
        
        ############################################################
        # Save all predictions into json file, make correlation plot
        ############################################################
        save_colored_correlation_pred_vs_label_plot(predictions, 
            labels, 
            plot_title=split_name + ' Correlation Plot', 
            plot_save_name=split_name + '_correlation_plot', 
            SAVE_PATH=SAVE_PATH)
        plot_continent_pred_vs_ground_truth_trend(predictions, labels, split_name, SAVE_PATH=SAVE_PATH)
        plot_continent_CDFs(predictions, labels, split_name, SAVE_PATH=SAVE_PATH)

        # Cast tensors to lists so that they are json serializable
        for i in range(len(predictions)):
            predictions[i] = [float(num) for num in predictions[i]]
            labels[i] = [float(num) for num in labels[i]]
        
        test_results = {
            "test_predictions": predictions,
            "test_labels": labels
        }
        with open(os.path.join(SAVE_PATH, "test_predictions_save.json"), "w") as test_f:
            json.dump(test_results, test_f, indent=4)

"""
matr = np.zeros((10, 10))
for i in range(alphas_edge_index1.shape[1]):
    matr[alphas_edge_index1[0,i], alphas_edge_index1[1,i]] = attention_weights1[i].mean()
"""
