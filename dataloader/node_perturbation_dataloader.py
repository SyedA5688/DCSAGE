import torch
import numpy as np
from torch.utils.data import Dataset


class Covid10CountriesUnperturbedDataset(Dataset):
    """
    This dataloader was written by Syed to make loading spatiotemporal 1-day graph datasets
    easier in Pytorch and Pytorch Geometric. The dataloader returns numpy matrices of the 
    dataset features and flight edge information, and the Pytorch Geometric graph object 
    is created in the main loop of the training function

    Since Numpy doesn't support jagged arrays easily, edge_idx_array will be initialized to 
    shape (2, max_num_edges) with -1s. Edge connection indices for each day are filled in, 
    meaning that edge_idx_array will contain edge information with -1s padding the array to
    make it to a consistent length.

    Args:
    - dataset_npz_path: Path to numpy zip npz file containing node feature information
    - window_size: Number of previous days to feed into network before predicting next day.
    - split: Portion of dataset to load for this dataset object (default for node perturbation: entire-dataset-smooth)
    - avg_graph_structure: Whether or not to load averaged graph edges over the window

    returns:
    - __getitem__() will return four things:
        1. Numpy matrix shape (window_size, num_nodes, num_features) of node feature data
        2. Numpy matrix shape (window_size, max_num_edges=100, num_edge_features) of edge attribute data
        3. Numpy matrix shape (window_size, 2, max_num_edges=100) of edge indices
        4. Numpy matrix shape (num_nodes,) of labels
    """
    def __init__(self, dataset_npz_path, window_size=14, data_split="entire-dataset-smooth", avg_graph_structure=False):
        # ============ Load data ============
        if data_split == "entire-dataset-smooth" and not avg_graph_structure:
            feature_matrix = np.load(dataset_npz_path)["feature_matrix_smooth"]
            flight_matrix = np.load(dataset_npz_path)["flight_matrix_log10_scaled"]
        elif data_split == "entire-dataset-smooth" and avg_graph_structure:
            feature_matrix = np.load(dataset_npz_path)["feature_matrix_smooth"]
            flight_matrix = np.load(dataset_npz_path)["flight_matrix_unscaled"]
        else:
            raise RuntimeError("Unknown dataset split selected")
        
        assert feature_matrix.shape[0] == flight_matrix.shape[0], "Node feature and edge attribute matrices do not match"

        # ============ Create sliding window dataset ============
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
                    # Log base 10 scaling is done after averaging graph structure if avg_graph_structure is set to true
                    smoothened_window_flight_matrix = flight_matrix[day_idx: day_idx + window_size].mean(axis=0)
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
                
                edges_attr_array = np.expand_dims(edges_attr_array, axis=-1)  # pad with -1s

                window_node_feat.append(np.expand_dims(node_data, axis=0))
                window_edge_idx.append(np.expand_dims(edges_idx_array, axis=0))
                window_edge_attr.append(np.expand_dims(edges_attr_array, axis=0))
            
            window_node_feat = np.concatenate(window_node_feat, axis=0)
            window_edge_idx = np.concatenate(window_edge_idx, axis=0)
            window_edge_attr = np.concatenate(window_edge_attr, axis=0)
            window_labels = feature_matrix[day_idx + window_size, :, 1]

            # Append to all_window lists being created
            all_window_node_feat.append(np.expand_dims(window_node_feat, axis=0))
            all_window_edge_attr.append(np.expand_dims(window_edge_attr, axis=0))
            all_window_edge_idx.append(np.expand_dims(window_edge_idx, axis=0))
            all_window_labels.append(np.expand_dims(window_labels, axis=0))
        
        self.all_window_node_feat = np.concatenate(all_window_node_feat, axis=0)  # shape [num_windows, window_len, num_nodes, num_features]
        self.all_window_edge_attr = np.concatenate(all_window_edge_attr, axis=0)  # shape [num_windows, window_len, num_features, max_num_edges]
        self.all_window_edge_idx = np.concatenate(all_window_edge_idx, axis=0)  # shape [num_windows, window_len, max_num_edges, num_edge_features]
        self.all_window_labels = np.concatenate(all_window_labels, axis=0)  # shape (num_windows, num_nodes)

    def __len__(self):
        return len(self.all_window_labels)
    
    def __getitem__(self, index):
        window_node_feat = self.all_window_node_feat[index].astype(np.float32)
        window_edge_idx = self.all_window_edge_idx[index]
        window_edge_idx = torch.LongTensor(window_edge_idx)
        window_edge_attr = self.all_window_edge_attr[index]
        window_labels = self.all_window_labels[index]
        
        return window_node_feat[:,:,1:2], window_edge_idx, window_edge_attr, window_labels


class Covid10CountriesPerturbedDataset(Dataset):
    def __init__(self, dataset_npz_path, window_size=14, data_split="train", perturb_country_idx=0, avg_graph_structure=False):
        # ============ Load data ============
        if data_split == "entire-dataset-smooth" and not avg_graph_structure:
            feature_matrix = np.load(dataset_npz_path)["feature_matrix_smooth"]
            flight_matrix = np.load(dataset_npz_path)["flight_matrix_log10_scaled"]
        elif data_split == "entire-dataset-smooth" and avg_graph_structure:
            feature_matrix = np.load(dataset_npz_path)["feature_matrix_smooth"]
            flight_matrix = np.load(dataset_npz_path)["flight_matrix_unscaled"]
        else:
            raise RuntimeError("Unknown dataset split selected")
        
        assert feature_matrix.shape[0] == flight_matrix.shape[0], "Node feature and edge attribute matrices do not match"
        
        # ============ Perturb chosen country by deleting edges ============
        flight_matrix[:, perturb_country_idx, :] = 0
        flight_matrix[:, :, perturb_country_idx] = 0

        self.feature_matrix = feature_matrix
        self.flight_matrix = flight_matrix

        # ============ Create sliding window dataset ============
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
                    # Log base 10 scaling is done after averaging graph structure if avg_graph_structure is set to true
                    smoothened_window_flight_matrix = flight_matrix[day_idx: day_idx + window_size].mean(axis=0)
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
                
                edges_attr_array = np.expand_dims(edges_attr_array, axis=-1)

                window_node_feat.append(np.expand_dims(node_data, axis=0))
                window_edge_idx.append(np.expand_dims(edges_idx_array, axis=0))
                window_edge_attr.append(np.expand_dims(edges_attr_array, axis=0))
            
            window_node_feat = np.concatenate(window_node_feat, axis=0)
            window_edge_idx = np.concatenate(window_edge_idx, axis=0)
            window_edge_attr = np.concatenate(window_edge_attr, axis=0)
            window_labels = feature_matrix[day_idx + window_size, :, 1]

            all_window_node_feat.append(np.expand_dims(window_node_feat, axis=0))
            all_window_edge_attr.append(np.expand_dims(window_edge_attr, axis=0))
            all_window_edge_idx.append(np.expand_dims(window_edge_idx, axis=0))
            all_window_labels.append(np.expand_dims(window_labels, axis=0))
        
        self.all_window_node_feat = np.concatenate(all_window_node_feat, axis=0)  # shape [num_windows, window_len, num_nodes, num_features]
        self.all_window_edge_attr = np.concatenate(all_window_edge_attr, axis=0)  # shape [num_windows, window_len, num_features, max_num_edges]
        self.all_window_edge_idx = np.concatenate(all_window_edge_idx, axis=0)  # shape [num_windows, window_len, max_num_edges, num_edge_features]
        self.all_window_labels = np.concatenate(all_window_labels, axis=0)  # shape (num_windows, num_nodes)

    def __len__(self):
        return len(self.all_window_labels)
    
    def __getitem__(self, index):
        window_node_feat = self.all_window_node_feat[index].astype(np.float32)
        window_edge_idx = self.all_window_edge_idx[index]
        window_edge_idx = torch.LongTensor(window_edge_idx)
        window_edge_attr = self.all_window_edge_attr[index]
        window_labels = self.all_window_labels[index]
        
        return window_node_feat[:,:,1:2], window_edge_idx, window_edge_attr, window_labels
