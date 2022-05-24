import os
import time
import json
import multiprocessing as mp
from random import seed

import torch
import numpy as np
from torch.utils.data import DataLoader

from util import *
from networks.dcsage import DynamicAdjSAGE
from networks.dcsage_gru import DCSAGE_GRU
from networks.dcsage_v2 import DCSAGE_v2
from networks.dcgat import DCGAT
from networks.dcgcn import DCGCN
from networks.dcgin import DCGIN
from networks.dcsage_temporal_attn import DCSAGE_Temporal_Attn
from covid_10country_perturb_dataset import Covid10CountriesUnperturbedDataset, Covid10CountriesPerturbedDataset
from node_perturbation_analysis import generate_perturbed_recursive_pred_dfs, generate_regular_recursive_pred_df

chosen_seed = 0
seed(chosen_seed)
np.random.seed(chosen_seed)
torch.manual_seed(chosen_seed)


def get_rolling_window_preds(args, SAVE_PATH, perturbed_dataloaders, unperturbed_dataloader, model_idx):
    """
    This function takes in the list of perturbed and unperturbed dataloaders as well as a model index.
    The function will first load the model corresponding to the model_idx that was saved during training.
    It will then apply this model to rolling window perturbation prediction saving by calling reused
    functions from node_perturbation_analysis.py.

    This function allows for parallelization of node perturbation analysis, because this function will
    apply only 1 model across all rolling windows.

    Args:
        - args:                     Arguments dictionary
        - SAVE_PATH:                Save directory
        - perturbed_dataloaders:    List of 10 perturbed dataloaders
        - unperturbed dataloader:   1 unperturbed dataloader
        - model idx:                Index of model to load and do perturbation analysis with
    """
    print("Process", os.getpid(), ", Model", model_idx)
    ############
    # Load model
    ############
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

    checkpoint = torch.load(os.path.join(args["training_path"], args["training_run"], "model_" + str(model_idx) + ".pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    ##############################################
    # Get predictions for one model on all windows
    ##############################################
    recur_pred_window = args["recursive_pred_len"]
    roll_win_one_model_pert_list = []
    roll_win_one_model_unpert_list = []
    
    for roll_win_idx in range(len(unperturbed_dataloader.dataset) - recur_pred_window - args["window"]):
        # print("Processing rolling window idx", roll_win_idx)
        perturb_df_nested_lists = generate_perturbed_recursive_pred_dfs(
            model, perturbed_dataloaders, args, SAVE_PATH, 
            model_idx=model_idx, 
            overwrite_window_idx=roll_win_idx,
            save_plot=False)  # (10, 10, 30, 4) - pert, country, len_rec_pred, columns
        
        regular_df_nested_lists = generate_regular_recursive_pred_df(
            model, unperturbed_dataloader, args, SAVE_PATH, 
            model_idx=model_idx, 
            overwrite_window_idx=roll_win_idx,
            save_plot=False)  # (10, 30, 4) - country, len_rec_pred, columns

        roll_win_one_model_pert_list.append(perturb_df_nested_lists)
        roll_win_one_model_unpert_list.append(regular_df_nested_lists)

    return (model_idx, roll_win_one_model_pert_list, roll_win_one_model_unpert_list)


model_process_results = []
def collect_results(result):
    global model_process_results
    model_process_results.append(result)


def main():
    """
    This function is the main driver of the program. It will create 10 perturbed dataloaders (1 for each country
    in graph that is perturbed), as well as 1 unperturbed dataloader. It will then create a Python multiprocessing
    pool to parallelize the node perturbation code for multiple graph models.

    One process will be created for each model that we want to run node perturbation on, and the process will apply
    this model to get prediction across all rolling windows. A callback function will be called when each process
    completed to collect the results from child processes, and once all child processes are finished the collected
    results (which may be out of order, since we use multiprocessing.apply_async() here) will be sorted to be in
    model order. The resulting array here will be shape (482, 100, 10, 10, 30, 4) for roll_win_pert_pred_nested_list 
    and (482, 100, 10, 30, 4) for roll_win_unpert_pred_nested_list.

    Once roll_win_pert_pred_nested_list and roll_win_unpert_pred_nested_list have been reassembled, then the 
    remainder of this function will create save directories and save the two large arrays, and then print out
    the total runtime of this script.
    """
    print("Parent process", os.getpid())

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
        avg_graph_structure=args["avg_graph_structure"])
    unperturbed_dataloader = DataLoader(unperturbed_dataset, batch_size=args['batch_size'], shuffle=False)
    
    total_start_time = time.time()

    ##########################################################
    # Multiprocessing, each model becomes one separate process
    ##########################################################
    num_cpus = mp.cpu_count() - 2
    with mp.Pool(processes=num_cpus) as pool:
        for model_idx in range(args["num_models"]):
            pool.apply_async(
                get_rolling_window_preds, 
                args=(args, SAVE_PATH, perturbed_dataloaders, unperturbed_dataloader, model_idx), 
                callback=collect_results)
        pool.close()
        pool.join()
    # get_rolling_window_preds(args, SAVE_PATH, perturbed_dataloaders, unperturbed_dataloader, 1)

    # Callback has collected results from child processes, sort now by model index in tuple
    model_process_results.sort(key=lambda x: x[0])  # Sorting by model index in tuple. Model_process_results is (100, tuple of (model_idx, pert_list, unpert_list))
    roll_win_pert_pred_nested_list = []
    roll_win_unpert_pred_nested_list = []
    
    for roll_win_idx in range(len(unperturbed_dataloader.dataset) - args["recursive_pred_len"] - args["window"]):
        model_pert_sublist = []  # Make (100, 10, 10, 30, 4) for this rolling window
        model_unpert_sublist = []  # Make (100, 10, 30, 4) for this rolling window
        for model_idx in range(args["num_models"]):
            model_pert_sublist.append(model_process_results[model_idx][1][roll_win_idx])
            model_unpert_sublist.append(model_process_results[model_idx][2][roll_win_idx])
        roll_win_pert_pred_nested_list.append(model_pert_sublist)
        roll_win_unpert_pred_nested_list.append(model_unpert_sublist)
    
    os.mkdir(os.path.join(SAVE_PATH, "prediction_saves"))
    np.save(os.path.join(SAVE_PATH, "prediction_saves", "roll_win_pert_preds.npy"), np.array(roll_win_pert_pred_nested_list))
    np.save(os.path.join(SAVE_PATH, "prediction_saves", "roll_win_unpert_preds.npy"), np.array(roll_win_unpert_pred_nested_list))

    print("\nTotal multiple-model node perturbation analysis took {:.3f} minutes".format((time.time() - total_start_time) / 60.))


if __name__ == "__main__":
    mp.set_start_method('spawn')
    with open("/Users/syedrizvi/Desktop/Projects/GNN_Project/DCSAGE/Node-Perturbation/node_perturb_analysis_config.json", "r") as f:
        args = json.load(f)
    args["save_dir"] = "/Users/syedrizvi/Desktop/Projects/GNN_Project/DCSAGE/Node-Perturbation/analysis-runs-multiple-models"
    args["num_models"] = 30  # 100

    if not os.path.exists(args["save_dir"]):
        os.mkdir(args["save_dir"])
    
    date = args["training_run"]
    SAVE_PATH = os.path.join(args["save_dir"], date)
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
        os.mkdir(os.path.join(SAVE_PATH, "python_file_saves"))
    
    os.system("cp node_perturb_analysis_config.json {}".format(os.path.join(SAVE_PATH, "python_file_saves")))
    os.system("cp node_perturbation_analysis.py {}".format(os.path.join(SAVE_PATH, "python_file_saves")))
    os.system("cp multiple_model_pert_analysis.py {}".format(os.path.join(SAVE_PATH, "python_file_saves")))
    os.system("cp covid_10country_perturb_dataset.py {}".format(os.path.join(SAVE_PATH, "python_file_saves")))
    os.system("cp networks/weight_sage.py {}".format(os.path.join(SAVE_PATH, "python_file_saves")))

    main()
