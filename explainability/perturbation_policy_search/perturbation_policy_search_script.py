import os
import time
import itertools
import multiprocessing as mp

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from typing import Union, Tuple
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptPairTensor, Adj, Size

from dcsage import DynamicAdjSAGE


class Covid10CountriesPerturbedDataset(Dataset):
    def __init__(self, dataset_npz_path, SENSITIVITY_ORDER_IDX, window_size=7, data_split="entire-dataset-smooth", avg_graph_structure=False, perturb_node_steps=[-0.25,0,0,0,0,0,0,0,0,0]):
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
        
        ################################################################
        # Perturb chosen country by deleting incoming and outgoing edges
        ################################################################
        for node_idx, perturb_step in zip(SENSITIVITY_ORDER_IDX, perturb_node_steps):
            flight_matrix[:, node_idx, :] *= (1 + perturb_step)
            flight_matrix[:, :, node_idx] *= (1 + perturb_step)

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
        
        return window_node_feat[:,:,1:2], window_edge_idx, window_edge_attr, window_labels


#######################
# Parallelized function
#######################
def perturbation_policy_search(args, all_dataloaders, model_idx):
    # Load model
    print("Process {}, Model {}".format(os.getpid(), model_idx))
    model = DynamicAdjSAGE(node_features=1, emb_dim=10, window_size=args["WINDOW_SIZE"], output=1, training=False, lstm_type="vanilla", name="DCSAGE")
    checkpoint = torch.load(os.path.join(args["TRAINING_EXPT_DIR"], args["TRAINING_RUN"], "model_{}.pth".format(model_idx)))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    model_predictions = []
    with torch.no_grad():
        for idx, dataloader in enumerate(all_dataloaders):
            perturbation_predictions = []

            bound = len(dataloader.dataset) - args["RECURSIVE_WINDOW_LENGTH"] - args["WINDOW_SIZE"]
            for overwrite_window_idx in range(0, bound):
                ##############################
                # Run recursive testing loop #
                ##############################
                extended_window_ncases = torch.zeros(args["RECURSIVE_WINDOW_LENGTH"] + args["WINDOW_SIZE"], 10)  # [37, 10]
                assert dataloader.dataset[overwrite_window_idx][0].shape[2] == 1, "Recursive prediction code made for 1 node feat currently"
                extended_window_ncases[0:args["WINDOW_SIZE"],:] = torch.from_numpy(dataloader.dataset[overwrite_window_idx][0][:,:,0])

                for batch_window_node_feat, batch_window_edge_idx, batch_window_edge_attr, batch_window_labels in dataloader:

                    for count, window_idx in enumerate(range(overwrite_window_idx, overwrite_window_idx + args["RECURSIVE_WINDOW_LENGTH"])):
                        window_node_feat = batch_window_node_feat[window_idx]
                        window_edge_idx = batch_window_edge_idx[window_idx]
                        window_edge_attr = batch_window_edge_attr[window_idx]
                        window_labels = batch_window_labels[window_idx]
                        
                        if count > 0:
                            window_node_feat[:,:,0] = extended_window_ncases[count:count + args["WINDOW_SIZE"]]

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
                            
                        extended_window_ncases[count + args["WINDOW_SIZE"], :] = y_hat[:,0]
                
                extended_window_ncases = extended_window_ncases[args["WINDOW_SIZE"]:,:].numpy()  # [30, 10]
                perturbation_predictions.append(extended_window_ncases)
            model_predictions.append(perturbation_predictions)
    
    # Return model_predictions and model_idx, so that model results can be sorted back in model order later
    return (model_idx, model_predictions)



#######################################
# Parallel processing result collection
#######################################
all_model_results = []
def collect_results(result):
    global all_model_results
    all_model_results.append(result)


def main():
    #########################
    # Define Global Variables
    #########################
    CONTINENTS = ["Africa", "North America", "South America", "Oceania", "Eastern Europe", "Western Europe", "Middle East", "South Asia", "Southeast-East Asia", "Central Asia"]
    TRAINING_EXPT_DIR = "/Users/syedrizvi/Desktop/Projects/GNN_Project/DCSAGE/Training-Code/training-runs-multiple-models/"
    TRAINING_RUN = "2022-04-16-00_41_22"  # DCSAGE 7-day 100 models 1 feature
    WINDOW_SIZE = 7
    RECURSIVE_WINDOW_LENGTH = 30

    NODES_TO_PERTURB = 5  # 3 means that 1, 2, or 3 nodes might be perturbed
    MODEL_IDXS = list(range(10))  # list(range(80, 100))  # List of model indices to use in analysis

    SENSITIVITY_ORDER = ["Western Europe", "North America", "Middle East", "Eastern Europe", "Southeast-East Asia", "Oceania", "South America", "South Asia", "Africa", "Central Asia"]
    SENSITIVITY_ORDER_IDX = [CONTINENTS.index(SENSITIVITY_ORDER[i]) for i in range(len(SENSITIVITY_ORDER))]
    PERTURBATION_STEPS = [-0.25, -0.5, -0.75]

    # Define save paths, create directory if doesn't exist
    SAVE_PATH = "./runs"
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    SAVE_PATH = os.path.join(SAVE_PATH, TRAINING_RUN, "models{}-{}".format(MODEL_IDXS[0], MODEL_IDXS[-1]))
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    #########################################################################
    # Define arguments object, so that child processes can use one arg object
    #########################################################################
    args = {
        "CONTINENTS": CONTINENTS,
        "TRAINING_EXPT_DIR": TRAINING_EXPT_DIR,
        "TRAINING_RUN": TRAINING_RUN,
        "WINDOW_SIZE": WINDOW_SIZE,
        "RECURSIVE_WINDOW_LENGTH": RECURSIVE_WINDOW_LENGTH,
        "NODES_TO_PERTURB": NODES_TO_PERTURB,
        "MODEL_IDXS": MODEL_IDXS,
        "SENSITIVITY_ORDER": SENSITIVITY_ORDER,
        "SENSITIVITY_ORDER_IDX": SENSITIVITY_ORDER_IDX,
        "PERTURBATION_STEPS": PERTURBATION_STEPS,
        "SAVE_PATH": SAVE_PATH
    }

    ####################
    # Create dataloaders
    ####################
    all_dataloaders = []

    unpert_dataset = Covid10CountriesPerturbedDataset(
        dataset_npz_path="./10_continents_dataset_v19_node_pert.npz",
        SENSITIVITY_ORDER_IDX=SENSITIVITY_ORDER_IDX,
        window_size=WINDOW_SIZE,
        data_split="entire-dataset-smooth",
        avg_graph_structure=False,
        perturb_node_steps=[0,0,0,0,0,0,0,0,0,0]
    )
    unpert_dataloader = DataLoader(unpert_dataset, batch_size=800, shuffle=False)
    all_dataloaders.append(unpert_dataloader)

    for num_nodes_to_pert in range(1, NODES_TO_PERTURB + 1):
        combinations_list = [p for p in itertools.product(PERTURBATION_STEPS, repeat=num_nodes_to_pert)]

        for combo_tuple in combinations_list:
            perturb_node_steps = [0,0,0,0,0,0,0,0,0,0]
            for idx, val in enumerate(combo_tuple):
                perturb_node_steps[idx] = val

            dataset = Covid10CountriesPerturbedDataset(
                dataset_npz_path="./10_continents_dataset_v19_node_pert.npz",
                SENSITIVITY_ORDER_IDX=SENSITIVITY_ORDER_IDX,
                window_size=WINDOW_SIZE,
                data_split="entire-dataset-smooth",
                avg_graph_structure=False,
                perturb_node_steps=perturb_node_steps
            )
            dataloader = DataLoader(dataset, batch_size=800, shuffle=False)
            all_dataloaders.append(dataloader)
    
    print("Number of dataloaders: {}\n".format(len(all_dataloaders)))

    ##########################################################
    # Multiprocessing, each model becomes one separate process
    ##########################################################
    total_start_time = time.time()
    num_cpus = mp.cpu_count() - 5  # Using 4 CPUs for now
    with mp.Pool(processes=num_cpus) as pool:
        for model_idx in MODEL_IDXS:
            pool.apply_async(
                perturbation_policy_search, 
                args=(args, all_dataloaders, model_idx), 
                callback=collect_results)
        pool.close()
        pool.join()
    
    # For debugging
    # perturbation_policy_search(args, all_dataloaders, model_idx=0)
    print("\nChild processes finished. Total multiple-model training time was {:.3f} minutes".format((time.time() - total_start_time) / 60.))

    # Callback has collected results from child processes, sort now by model index in tuple
    all_model_results.sort(key=lambda x: x[0])

    # Aggregate into one array
    all_model_predictions = []
    for tup in all_model_results:
        all_model_predictions.append(tup[1])
    all_model_predictions_np = np.array(all_model_predictions)
    print("Shape of final array:", all_model_predictions_np.shape)

    # Save numpy array
    print("Saving array to save directory...")
    np.save(os.path.join(SAVE_PATH, "models_{}-{}_predictions".format(MODEL_IDXS[0], MODEL_IDXS[-1])), all_model_predictions_np)
    print("Done.\n")


if __name__ == "__main__":
    main()
