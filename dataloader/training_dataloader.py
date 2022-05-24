import torch
import numpy as np
from torch.utils.data import Dataset


class Covid10CountriesDataset(Dataset):
    """
    This dataset was written by Syed to overcome the limitations of Graph dataloaders from Pytorch
    Geometric and Pytorch Geometric Temporal that make it difficult to create and return graph 
    data/batch objects. This dataloader returns numpy matrices of the data, and the graph is created
    in the main loop of the training function

    *Note: Since numpy doesn't support jagged arrays well, edge_idx_array will be initialized to 
    shape (2, 100) with -1s filled in. Edge connections for each day are filled in, and the extra -1s
    need to be removed in code later

    Args:
    - dataset_npz_path:         Path to numpy zip npz file containing node feature information
    - window_size:              Number of previous days to feed into network before predicting next day.
    - split:                    Portion of dataset to load for this dataset object (default for node perturbation: entire-dataset-smooth)
    - avg_graph_structure:   Whether or not to load averaged graph edges over the window

    returns:
    - __getitem__() will return four things:
        1. Numpy matrix shape (window_size, 10, 3) of node feature data
        2. Numpy matrix shape (window_size, num_edges=100, 1) of edge attribute data
        3. Numpy matrix shape (window_size, 2, num_edges=100) of edge indices
        4. Numpy matrix shape (10,) of labels (number of cases in next day for each of the 10 nodes)
    """
    def __init__(self, dataset_npz_path, window_size=14, data_split="train", avg_graph_structure=False):
        if data_split == "train" and not avg_graph_structure:
            feature_matrix = np.load(dataset_npz_path)["train_features_log10"]
            flight_matrix = np.load(dataset_npz_path)["train_log10_scaled_flight_matrix"]
        elif data_split == "train" and avg_graph_structure:    
            feature_matrix = np.load(dataset_npz_path)["train_features_log10"]
            flight_matrix = np.load(dataset_npz_path)["train_unscaled_flight_matrix"]
        elif data_split == "validation" and not avg_graph_structure:
            feature_matrix = np.load(dataset_npz_path)["val_features_log10"]
            flight_matrix = np.load(dataset_npz_path)["val_log10_scaled_flight_matrix"]
        elif data_split == "validation" and avg_graph_structure:
            feature_matrix = np.load(dataset_npz_path)["val_features_log10"]
            flight_matrix = np.load(dataset_npz_path)["val_unscaled_flight_matrix"]
        elif data_split == "test-unsmooth" and not avg_graph_structure:
            feature_matrix = np.load(dataset_npz_path)["test_features_log10_unsmooth"]
            flight_matrix = np.load(dataset_npz_path)["test_log10_scaled_flight_matrix"]
        elif data_split == "test-unsmooth" and avg_graph_structure:
            feature_matrix = np.load(dataset_npz_path)["test_features_log10_unsmooth"]
            flight_matrix = np.load(dataset_npz_path)["test_unscaled_flight_matrix"]
        elif data_split == "test-smooth" and not avg_graph_structure:
            feature_matrix = np.load(dataset_npz_path)["test_features_log10_smooth"]
            flight_matrix = np.load(dataset_npz_path)["test_log10_scaled_flight_matrix"]
        elif data_split == "test-smooth" and avg_graph_structure:
            feature_matrix = np.load(dataset_npz_path)["test_features_log10_smooth"]
            flight_matrix = np.load(dataset_npz_path)["test_unscaled_flight_matrix"]
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
        
        for day_idx in range(0, len(feature_matrix) - window_size - 1):
            window_node_feat = []
            window_edge_idx = []
            window_edge_attr = []

            for sub_day in range(day_idx, day_idx + window_size):
                node_data = feature_matrix[sub_day]
                edges_idx_array = np.full((2, 100), -1)
                edges_attr_array = np.full((100), -1).astype(np.float32)

                if avg_graph_structure:
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
                                edges_idx_array[0][edge_idx] = row
                                edges_idx_array[1][edge_idx] = col
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

                window_node_feat.append(np.expand_dims(node_data, axis=0))
                window_edge_idx.append(np.expand_dims(edges_idx_array, axis=0))
                window_edge_attr.append(np.expand_dims(edges_attr_array, axis=0))
            
            window_node_feat = np.concatenate(window_node_feat, axis=0)  # shape (14, 10, 2)
            window_edge_idx = np.concatenate(window_edge_idx, axis=0)  # Shape (14, 2, num_edges=100)
            window_edge_attr = np.concatenate(window_edge_attr, axis=0)  # Shape (14, num_edges=100, 1)
            window_labels = feature_matrix[day_idx + window_size, :, 1]  # Shape (10)

            all_window_node_feat.append(np.expand_dims(window_node_feat, axis=0))
            all_window_edge_attr.append(np.expand_dims(window_edge_attr, axis=0))
            all_window_edge_idx.append(np.expand_dims(window_edge_idx, axis=0))
            all_window_labels.append(np.expand_dims(window_labels, axis=0))
        
        self.all_window_node_feat = np.concatenate(all_window_node_feat, axis=0)
        self.all_window_edge_attr = np.concatenate(all_window_edge_attr, axis=0)
        self.all_window_edge_idx = np.concatenate(all_window_edge_idx, axis=0)
        self.all_window_labels = np.concatenate(all_window_labels, axis=0)  # shape (dataset_len, 10)

    def __len__(self):
        return len(self.all_window_labels)
    
    def __getitem__(self, index):
        window_node_feat = self.all_window_node_feat[index].astype(np.float32)  # shape (14, 10, 2)
        window_edge_idx = self.all_window_edge_idx[index]  # shape (14, 2, *num_edges=100)
        window_edge_idx = torch.LongTensor(window_edge_idx)
        window_edge_attr = self.all_window_edge_attr[index]  # shape (14, *num_edges=100, 1)
        window_labels = self.all_window_labels[index]  # shape (10)
        
        return window_node_feat[:,:,1:2], window_edge_idx, window_edge_attr, window_labels
