import os
import json
import torch

from utils.training_utils import *
from torch_geometric.data import Data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from models.dcsage import DynamicAdjSAGE
from models.dcsage_gru import DCSAGE_GRU
from models.dcgat import DCGAT
from models.dcgcn import DCGCN
from models.dcgin import DCGIN
from models.dcsage_temporal_attn import DCSAGE_Temporal_Attn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
This script applies test on an already trained model. Use this script if a model was trained
using train.py but for whatever reason did not complete the test evaluation at the end. This
script will generate the same figures and test results.
"""


class Covid10NodePerturbationCountriesDataset(Dataset):
    """
    Same dataloader as in node perturbation experiment. Loads the entire dataset.
    """
    def __init__(self, dataset_npz_path, window_size=14, data_split="entire-dataset-smooth", avg_graph_structure=False):
        ###########
        # Load data
        ###########
        if data_split == "entire-dataset-smooth" and not avg_graph_structure:
            feature_matrix = np.load(dataset_npz_path)["feature_matrix_smooth"]
            flight_matrix = np.load(dataset_npz_path)["flight_matrix_log10_scaled"]
        elif data_split == "entire-dataset-smooth" and avg_graph_structure:
            feature_matrix = np.load(dataset_npz_path)["feature_matrix_smooth"]
            flight_matrix = np.load(dataset_npz_path)["flight_matrix_unscaled"]
        else:
            raise RuntimeError("Unknown dataset split selected")
        
        assert feature_matrix.shape[0] == flight_matrix.shape[0], "Node feature and edge attribute matrices do not match"

        ###############################
        # Create sliding window dataset
        ###############################
        all_window_node_feat = []
        all_window_edge_attr = []
        all_window_edge_idx = []
        all_window_labels = []
        
        for day_idx in range(0, len(feature_matrix) - window_size):
            window_node_feat = []
            window_edge_idx = []
            window_edge_attr = []

            for sub_day in range(day_idx, day_idx + window_size):
                node_data = feature_matrix[sub_day]
                edges_idx_array = np.full((2, 100), -1)
                edges_attr_array = np.full((100), -1).astype(np.float32)

                if avg_graph_structure:
                    # Calculate smoothened graph structure across window
                    smoothened_window_flight_matrix = flight_matrix[day_idx: day_idx + window_size].mean(axis=0)  # Shape [10, 10]
                    # Log10 scale now that adjacency is averaged
                    for row in range(len(smoothened_window_flight_matrix)):
                        for col in range(len(smoothened_window_flight_matrix[row])):
                            if smoothened_window_flight_matrix[row][col] > 0:
                                smoothened_window_flight_matrix[row][col] = np.log10(smoothened_window_flight_matrix[row][col])
                    
                    edge_idx = 0
                    for row in range(len(smoothened_window_flight_matrix)):
                        for col in range(len(smoothened_window_flight_matrix[row])):
                            if smoothened_window_flight_matrix[row][col] > 0:
                                edges_idx_array[0][edge_idx] = row  # row is source node
                                edges_idx_array[1][edge_idx] = col  # col is dest node
                                edges_attr_array[edge_idx] = smoothened_window_flight_matrix[row][col]
                                edge_idx += 1
                else:
                    edge_idx = 0
                    for row in range(len(flight_matrix[sub_day])):
                        for col in range(len(flight_matrix[sub_day][row])):
                            if flight_matrix[sub_day][row][col] > 0:
                                edges_idx_array[0][edge_idx] = row
                                edges_idx_array[1][edge_idx] = col
                                edges_attr_array[edge_idx] = flight_matrix[sub_day][row][col]
                                edge_idx += 1
                
                edges_attr_array = np.expand_dims(edges_attr_array, axis=-1)  # --> shape (100, 1)

                # Append to single window lists
                window_node_feat.append(np.expand_dims(node_data, axis=0))
                window_edge_idx.append(np.expand_dims(edges_idx_array, axis=0))
                window_edge_attr.append(np.expand_dims(edges_attr_array, axis=0))
            
            
            # Single window of data calculated. Concatenate Python lists into numpy arrays
            window_node_feat = np.concatenate(window_node_feat, axis=0)  # shape (14, 10, 3)
            window_edge_idx = np.concatenate(window_edge_idx, axis=0)  # Shape (14, 2, *num_edges=100)
            window_edge_attr = np.concatenate(window_edge_attr, axis=0)  # Shape (14, *num_edges=100, 1)
            window_labels = feature_matrix[day_idx + window_size, :, 1]  # Shape (10)

            # Append to all_window lists being created
            all_window_node_feat.append(np.expand_dims(window_node_feat, axis=0))
            all_window_edge_attr.append(np.expand_dims(window_edge_attr, axis=0))
            all_window_edge_idx.append(np.expand_dims(window_edge_idx, axis=0))
            all_window_labels.append(np.expand_dims(window_labels, axis=0))
        
        self.all_window_node_feat = np.concatenate(all_window_node_feat, axis=0)  # shape (dataset_len, 14, 10, 3)
        self.all_window_edge_attr = np.concatenate(all_window_edge_attr, axis=0)  # shape (dataset_len, 14, num_edges=100, 1)
        self.all_window_edge_idx = np.concatenate(all_window_edge_idx, axis=0)  # shape (dataset_len, 14, 2, num_edges=100)
        self.all_window_labels = np.concatenate(all_window_labels, axis=0)  # shape (dataset_len, 10)

    def __len__(self):
        return len(self.all_window_labels)
    
    def __getitem__(self, index):
        window_node_feat = self.all_window_node_feat[index].astype(np.float32)  # shape (14, 10, 3)
        window_edge_idx = self.all_window_edge_idx[index]  # shape (14, 2, num_edges=100)
        window_edge_idx = torch.LongTensor(window_edge_idx)
        window_edge_attr = self.all_window_edge_attr[index]  # shape (14, num_edges=100, 1)
        window_labels = self.all_window_labels[index]  # shape (10)
        
        return window_node_feat[:,:,1:2], window_edge_idx, window_edge_attr, window_labels


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
            plot_spatial_attn_coeff_heatmap(window_alphas_attn_coeff1.mean(axis=0).mean(axis=0), title="DCSAGE 7-day GAT Layer 1 Mean Spatial Attention Heatmap", save_path=os.path.join(SAVE_PATH, "gat_layer1_mean_attn_heatmap.png"))
            plot_spatial_attn_coeff_heatmap(window_alphas_attn_coeff1.sum(axis=0).sum(axis=0), title="DCSAGE 7-day GAT Layer 1 Summed Spatial Attention Heatmap", save_path=os.path.join(SAVE_PATH, "gat_layer1_summed_attn_heatmap.png"))
            plot_spatial_attn_coeff_heatmap(window_alphas_attn_coeff2.mean(axis=0).mean(axis=0), title="DCSAGE 7-day GAT Layer 2 Mean Spatial Attention Heatmap", save_path=os.path.join(SAVE_PATH, "gat_layer2_mean_attn_heatmap.png"))
            plot_spatial_attn_coeff_heatmap(window_alphas_attn_coeff2.sum(axis=0).sum(axis=0), title="DCSAGE 7-day GAT Layer 2 Summed Spatial Attention Heatmap", save_path=os.path.join(SAVE_PATH, "gat_layer2_summed_attn_heatmap.png"))
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


if __name__ == "__main__":
    with open("./covid10countries_config.json", "r") as f:
        args = json.load(f)
    args["save_dir"] = "./training-runs"
    assert args["model_architecture"] in ["DCSAGE", "DCSAGE_v2", "DCGAT", "DCSAGE_Temporal_Attn"]

    SAVE_PATH = "/Users/syedrizvi/Desktop/Projects/GNN_Project/DCSAGE/Training-Code/training-runs/2022-04-16-10_49_25_GAT_Attn_7day_1feat"
    
    full_dataset = Covid10NodePerturbationCountriesDataset(
        dataset_npz_path="/Users/syedrizvi/Desktop/Projects/GNN_Project/DCSAGE/Training-Code/datasets/10_continents_dataset_v19_node_pert.npz",
        window_size=args['window'], 
        data_split="entire-dataset-smooth", 
        avg_graph_structure=args["avg_graph_structure"])

    full_dataset_dataloader = DataLoader(full_dataset, batch_size=args['batch_size'], shuffle=False)

    # Run test for unsmooth and smooth test set, training set, and validation set
    model_path = os.path.join(SAVE_PATH, "best_model.pth")
    os.mkdir(os.path.join(SAVE_PATH, "full_dataset"))
    test(model_path, args, full_dataset_dataloader, os.path.join(SAVE_PATH, "full_dataset"), split_name="Test_Unsmooth")

