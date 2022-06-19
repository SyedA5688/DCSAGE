import os
import time
import itertools
import multiprocessing as mp

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from models.dcsage import DynamicAdjSAGE
from dataloader.perturbation_policy_dataloader import Covid10CountriesPerturbedDataset


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
    TRAINING_EXPT_DIR = "./training/training-runs-multiple-models/"
    TRAINING_RUN = "2022-06-19-17_25_11"
    WINDOW_SIZE = 7
    RECURSIVE_WINDOW_LENGTH = 30

    NODES_TO_PERTURB = 5  # Combinations of 1 node will be perturbed, then combos of 2 nodes, ... up to NODES_TO_PERTURB
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
