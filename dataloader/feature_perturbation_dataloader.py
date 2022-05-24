import torch
import numpy as np
from torch.utils.data import Dataset


class CovidUnperturbedDataset(Dataset):
    def __init__(self, dataset_npz_path, window_size=7, data_split="entire-dataset-smooth", avg_graph_structure=False):
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
        
        return window_node_feat, window_edge_idx, window_edge_attr, window_labels


class CovidContainmentPerturbedDataset(Dataset):
    """
    Feature perturbation dataloader. Same as node perturbation dataloader, but instead of node edges being
    perturbed, containment index is set to either 0 or 100.
    """
    def __init__(self, dataset_npz_path, window_size=7, data_split="entire-dataset-smooth", containment_new_val=0, avg_graph_structure=False):
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
        
        ##########################################
        # Perturb by setting new containment index
        ##########################################
        feature_matrix[:, :, 0] = containment_new_val

        # Flight matrix is log10 scaled, so perturbation won't affect flight scaling
        self.feature_matrix = feature_matrix
        self.flight_matrix = flight_matrix

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
                
                edges_attr_array = np.expand_dims(edges_attr_array, axis=-1)

                window_node_feat.append(np.expand_dims(node_data, axis=0))
                window_edge_idx.append(np.expand_dims(edges_idx_array, axis=0))
                window_edge_attr.append(np.expand_dims(edges_attr_array, axis=0))
            
            window_node_feat = np.concatenate(window_node_feat, axis=0)  # shape (14, 10, 3)
            window_edge_idx = np.concatenate(window_edge_idx, axis=0)  # Shape (14, 2, *num_edges=100)
            window_edge_attr = np.concatenate(window_edge_attr, axis=0)  # Shape (14, *num_edges=100, 1)
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
        window_node_feat = self.all_window_node_feat[index].astype(np.float32)  # shape (14, 10, 3)
        window_edge_idx = self.all_window_edge_idx[index]  # shape (14, 2, *num_edges=100)
        window_edge_idx = torch.LongTensor(window_edge_idx)
        window_edge_attr = self.all_window_edge_attr[index]  # shape (14, *num_edges=100, 1)
        window_labels = self.all_window_labels[index]  # shape (10)
        
        return window_node_feat, window_edge_idx, window_edge_attr, window_labels


if __name__ == "__main__":
    dataset = CovidContainmentPerturbedDataset(dataset_npz_path="./10_continents_dataset_v19_node_pert.npz")
