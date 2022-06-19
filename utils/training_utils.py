import os
import torch
import statistics
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


"""
Loss function code obtained from https://github.com/jjgarau/GNND
Credit goes to authors: Nathan Sesti, Juan Jose Garau-Luis
"""

def mape_loss(output, label):
    return torch.mean(torch.abs(torch.div((output - label), label)))

def mse_loss(output, label, mean=None):
    return torch.mean(torch.square(output - label))

def msse_loss(output, label, mean=None):
    return torch.mean(torch.div(torch.square(output - label), label + 1))

def rmse_loss(output, label, mean=None):
    return torch.sqrt(torch.mean(torch.square(output - label)))

def mae_loss(output, label):
    return torch.mean(torch.abs(output - label))

def mase_loss(output, label, mean=None):
    mean = mean.reshape(output.shape)
    label_mean = torch.mean(label)
    if not mean is None:
        return torch.mean(torch.abs(output - label) / mean)
    elif label_mean == 0:
        return torch.mean(torch.abs(output - label))
    else:
        return torch.mean(torch.abs(output - label)) / label_mean


def mase1_loss(output, label, mean=None):
    # Extreme 1: all countries equal
    # L_i = (x_i - y_i)^2 / y_i
    # L = (L_1 + L_2 + … + L_N) / N
    label = label[:, 0]
    output = output.reshape(output.shape[0])
    label_mean = torch.mean(label)
    if not mean is None:
        return torch.mean(torch.abs(output - label) / mean)
    if label_mean == 0:
        return torch.mean(torch.abs(output - label))
    else:
        return torch.mean(torch.abs(output - label)) / label_mean


def mase2_loss(output, label, mean=None):  # mean is [10, 1]
    # Extreme 2: all people equal
    # X = (x_1 + x_2 + … + x_N)
    # Y = (y_1 + y_2 + … + y_N)
    # L = (X - Y)^2 / Y
    # label = label[:, 0]
    X = torch.sum(output)
    Y = torch.sum(label)
    if Y == 0 and not mean is None:
        return torch.abs(X - Y) / torch.sum(mean)
    elif Y == 0:
        return torch.abs(X - Y)
    else:
        # Training uses this conditional branch
        return torch.abs(X - Y) / Y


def anti_lag_loss(output, label, lagged_label, mean=None, loss_func=mase2_loss, penalty_factor=0.1):
    output = output.reshape(output.shape[0])
    lagged_label = lagged_label.reshape(lagged_label.shape[0])

    # Or instead of penalty factor (or with it) should I be using the same loss function and taking the inverse square of that to ensure good scaling?
    penalty = torch.mean(torch.div(1, torch.square(output - lagged_label)))

    return loss_func(output, label, mean=mean) + penalty * penalty_factor


def lag_factor(output, lagged_label):
    return torch.div(torch.abs(output - lagged_label), lagged_label)


def mase3_loss(output, label, populations, mean=None, k=500000):
    # Middle point: consider a population threshold k
    # x_k = sum(x_i) such that continent i has less than k population
    # y_k = sum(y_i) such that continent i has less than k population
    # L_i = (x_i - y_i)^2 / y_i   for countries i with more than k population
    # L_k = (x_k - y_k)^2 / y_k
    # L = L_k + sum(L_i)
    label = label[:, 0]

    if mean is None:
        mean = torch.mean(label)
    if sum(mean) == 0:
        mean = 1

    large_outputs = []
    large_labels = []
    large_means = []

    small_outputs = []
    small_labels = []
    small_means = []
    for i in range(len(populations)):
        if populations[i] < k:
            small_outputs.append(output[i])
            small_labels.append(label[i])
            small_means.append(mean[i])
        else:
            large_outputs.append(output[i])
            large_labels.append(label[i])
            large_means.append(mean[i])

    x_k = sum(small_outputs)
    y_k = sum(small_labels)
    L_i = torch.abs(torch.FloatTensor(large_outputs) - torch.FloatTensor(large_labels)) / torch.FloatTensor(large_means)
    L_k = abs(x_k - y_k) / sum(small_means)
    return L_k + torch.sum(L_i)


def inv_reg_mase_loss(output, label):
    return mase_loss(output, label) + torch.mean(torch.div(1, output))


######################################
# Plotting Code, written by Syed Rizvi
######################################

def save_colored_correlation_pred_vs_label_plot(model_preds, labels, plot_title, plot_save_name, SAVE_PATH):
    val_preds = [num.item() for pred in model_preds for num in pred]
    val_labels = [num.item() for label in labels for num in label]
    # country_categ = ["Brazil", "Germany", "Spain", "France", "Britain", "India", "Italy", "Russia", "Turkey", "USA"] * len(model_preds)
    continent_categ = ["Africa", "North America", "South America", "Oceania", "Eastern Europe", "Western Europe", "Middle East", "South Asia", "Southeast-East Asia", "Central Asia"] * len(model_preds)

    visual_df = pd.DataFrame({
        "Predictions": val_preds,
        "Ground Truth": val_labels,
        "Continent_Categ": continent_categ
    })
    corr_coeff, _ = pearsonr(x=np.array(val_preds), y=np.array(val_labels))

    max_num = max(max(val_preds), max(val_labels))
    max_num = int(max_num)

    x_points = [2, max_num / 2, max_num + 2]
    y_points = [2, max_num / 2, max_num + 2]

    plt.rcParams.update({'font.size': 16})
    sns.scatterplot(data=visual_df, x="Ground Truth", y="Predictions", hue="Continent_Categ")
    plt.plot(x_points, y_points, color="green", label="y=x")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlim(2, max_num + 0.6)
    plt.ylim(2, max_num + 0.6)
    plt.title(plot_title + " (Correlation: {:.2f})".format(corr_coeff), fontsize=18)
    plt.savefig(SAVE_PATH + "/" + plot_save_name + ".png", bbox_inches='tight')
    plt.clf()
    plt.close()


def save_colored_correlation_pred_vs_label_plot_multistep(model_preds, labels, plot_title, plot_save_name, SAVE_PATH):
    val_preds = []
    val_labels = []
    continent_categ = []
    continents = ["Africa", "North America", "South America", "Oceania", "Eastern Europe", "Western Europe", "Middle East", "South Asia", "Southeast-East Asia", "Central Asia"] 

    for sample_idx in range(len(model_preds)):
        for pred_idx in range(len(model_preds[sample_idx])):
            for node_idx in range(len(model_preds[sample_idx][pred_idx])):
                val_preds.append(model_preds[sample_idx][pred_idx, node_idx])
                val_labels.append(labels[sample_idx][pred_idx, node_idx])
                continent_categ.append(continents[node_idx])

    visual_df = pd.DataFrame({
        "Predictions": val_preds,
        "Ground Truth": val_labels,
        "Continent_Categ": continent_categ
    })

    max_num = max(max(val_preds), max(val_labels))
    max_num = int(max_num)

    x_points = [0, max_num / 2, max_num + 2]
    y_points = [0, max_num / 2, max_num + 2]

    plt.rcParams.update({'font.size': 16})
    sns.scatterplot(data=visual_df, x="Ground Truth", y="Predictions", hue="Continent_Categ")
    plt.plot(x_points, y_points, color="green", label="y=x")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlim(2, max_num + 1)
    plt.ylim(2, max_num + 1)
    plt.title(plot_title)
    plt.savefig(SAVE_PATH + "/" + plot_save_name + ".png", bbox_inches='tight')
    plt.clf()
    plt.close()


def plot_continent_pred_vs_ground_truth_trend(model_preds, labels, split_name, SAVE_PATH):
    continents = ["Africa", "North America", "South America", "Oceania", "Eastern Europe", "Western Europe", "Middle East", "South Asia", "Southeast-East Asia", "Central Asia"]
    
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(25,10))
    plt.rcParams.update({'font.size': 18})
    fig.suptitle("DCSAGE " + split_name + " Set Regular Pred Per Continent")
    
    idx = 0
    for row in ax:
        for col in row:
            continent_preds = []
            continent_labels = []
            for i in range(len(model_preds)):
                continent_preds.append(float(model_preds[i][idx]))
                continent_labels.append(float(labels[i][idx]))

            visual_df = pd.DataFrame({
            "Predictions": continent_preds,
            "Ground Truth": continent_labels,
            "Day Index": list(range(len(continent_preds))) 
            })

            sns.lineplot(ax=col, x='Day Index', y='Cases', hue='Prediction Type', data=pd.melt(visual_df, ['Day Index'], value_name="Cases", var_name="Prediction Type"))
            col.set_title(continents[idx])
            col.set_ylim([0.5, 5.5])
            idx += 1

    plt.tight_layout()
    plt.savefig(SAVE_PATH + '/reg_preds.png', bbox_inches='tight')
    plt.clf()
    plt.close()

def plot_continent_pred_vs_ground_truth_trend_multistep(model_preds, labels, split_name, SAVE_PATH):
    continents = ["Africa", "North America", "South America", "Oceania", "Eastern Europe", "Western Europe", "Middle East", "South Asia", "Southeast-East Asia", "Central Asia"]
    
    for sample_idx in range(len(model_preds)):
        fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(18,10))
        plt.rcParams.update({'font.size': 18})
        fig.suptitle("DCSAGE " + split_name + " Set Regular Pred Per Continent")
        
        idx = 0
        for row in ax:
            for col in row:
                continent_preds = []
                continent_labels = []
                for i in range(len(model_preds[sample_idx])):
                    continent_preds.append(float(model_preds[sample_idx][i][idx]))
                    continent_labels.append(float(labels[sample_idx][i][idx]))

                visual_df = pd.DataFrame({
                "Predictions": continent_preds,
                "Ground Truth": continent_labels,
                "Day Index": list(range(len(continent_preds))) 
                })

                sns.lineplot(ax=col, x='Day Index', y='value', hue='variable', data=pd.melt(visual_df, ['Day Index']))
                col.set_title(continents[idx])
                col.set_ylim([0.5, 6])
                idx += 1

        plt.tight_layout()
        plt.savefig(SAVE_PATH + '/reg_preds_sample_{}.png'.format(sample_idx), bbox_inches='tight')
        plt.clf()
        plt.close()


def plot_continent_CDFs(model_preds, labels, split_name, SAVE_PATH):
    continents = ["Africa", "North America", "South America", "Oceania", "Eastern Europe", "Western Europe", "Middle East", "South Asia", "Southeast-East Asia", "Central Asia"]
    
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(25,12))
    plt.rcParams.update({'font.size': 18})
    fig.suptitle("DCSAGE " + split_name + " Set CDF Plots Per Continent")

    idx = 0
    for row in ax:
        for col in row:
            continent_preds = []
            continent_labels = []
            for i in range(len(model_preds)):
                continent_preds.append(float(model_preds[i][idx]))
                continent_labels.append(float(labels[i][idx]))

            visual_df = pd.DataFrame({
            "Predictions": continent_preds,
            "Ground Truth": continent_labels,
            })

            continent_pred_stddev = statistics.stdev(continent_preds)

            sns.ecdfplot(ax=col, data=pd.melt(visual_df, value_name="Log10 Number of Cases"), x='Log10 Number of Cases', hue='variable')
            col.set_title(continents[idx] + "\nStd: " + str(round(continent_pred_stddev, 4)))
            col.set_xlim([0.5, 5.5])
            idx += 1

    plt.tight_layout()
    plt.savefig(SAVE_PATH + '/cdf_plots.png', bbox_inches='tight')
    plt.clf()
    plt.close()


def plot_continent_CDF_multistep(model_preds, labels, split_name, SAVE_PATH):
    continents = ["Africa", "North America", "South America", "Oceania", "Eastern Europe", "Western Europe", "Middle East", "South Asia", "Southeast-East Asia", "Central Asia"]
    
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(25,10))
    plt.rcParams.update({'font.size': 18})
    fig.suptitle("DCSAGE " + split_name + " Set CDF Plots Per Continent")

    idx = 0
    for row in ax:
        for col in row:
            continent_preds = []
            continent_labels = []
            for sample_idx in range(len(model_preds)):
                for i in range(len(model_preds[sample_idx])):
                    continent_preds.append(float(model_preds[sample_idx][i,idx]))
                    continent_labels.append(float(labels[sample_idx][i,idx]))

            visual_df = pd.DataFrame({
            "Predictions": continent_preds,
            "Ground Truth": continent_labels,
            })

            continent_pred_stddev = statistics.stdev(continent_preds)

            sns.ecdfplot(ax=col, data=pd.melt(visual_df, value_name="Log10 Number of Cases"), x='Log10 Number of Cases', hue='variable')
            col.set_title(continents[idx] + ", Std: " + str(round(continent_pred_stddev, 4)))
            col.set_xlim([0.5, 5.5])
            idx += 1

    plt.tight_layout()
    plt.savefig(SAVE_PATH + '/cdf_plots.png', bbox_inches='tight')
    plt.clf()
    plt.close()


def plot_loss_curves(train_losses, val_losses, epoch_count, SAVE_PATH, model_num=None):
    time = list(range(epoch_count))
    visual_df = pd.DataFrame({
        "Train Loss": train_losses,
        "Validation Loss": val_losses,
        "Epoch": time
    })

    plt.rcParams.update({'font.size': 16})
    sns.lineplot(x='Epoch', y='Loss Value', hue='Loss Type', data=pd.melt(visual_df, ['Epoch'], value_name="Loss Value", var_name="Loss Type"))
    plt.title("DCSAGE Loss Curves")
    filename = "train_val_loss_curves" if model_num == None else "loss_curves_model_" + str(model_num)
    plt.savefig(os.path.join(SAVE_PATH, filename + '.png'), bbox_inches='tight')
    plt.clf()
    plt.close()


def plot_continent_pred_vs_ground_truth_vs_extended_feedback_pred_trend(model_preds, labels, extended_feedback_preds, SAVE_PATH):
    assert len(model_preds) == len(labels) == len(extended_feedback_preds), "Trends have different lengths"
    continents = ["Africa", "North America", "South America", "Oceania", "Eastern Europe", "Western Europe", "Middle East", "South Asia", "Southeast-East Asia", "Central Asia"]
    
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(18,10))
    plt.rcParams.update({'font.size': 18})
    fig.suptitle("DCSAGE Recursive, Regular Prediction, and Ground Truth Trends")

    idx = 0
    for row in ax:
        for col in row:
            continent_preds = []
            continent_labels = []
            continent_extended_feedback_preds = []
            for i in range(len(model_preds)):
                continent_preds.append(float(model_preds[i][idx]))
                continent_labels.append(float(labels[i][idx]))
                continent_extended_feedback_preds.append(float(extended_feedback_preds[i][idx]))

            visual_df = pd.DataFrame({
            "Regular Predictions": continent_preds,
            "Ground Truth": continent_labels,
            "Extended Recursive Predictions": continent_extended_feedback_preds,
            "Day Index": list(range(len(continent_preds))) 
            })

            sns.lineplot(ax=col, x='Day Index', y='value', hue='variable', data=pd.melt(visual_df, ['Day Index']))
            col.set_title(continents[idx])
            col.set_ylim([0.5, 5.5])
            idx += 1

    plt.savefig(SAVE_PATH + '/test_set_trends.png', bbox_inches='tight')
    plt.clf()
    plt.close()


def plot_tsne_reduced_embeddings(reduced_embeddings_matrix, plot_title, save_filename, SAVE_PATH):
    """
    reduced_embeddings_matrix: Numpy matrix of shape [dataset_len, 10, 2]
    """
    reshaped_reduced_emb = np.moveaxis(reduced_embeddings_matrix, 0, 1)
    continents = ["Africa", "North America", "South America", "Oceania", "Eastern Europe", "Western Europe", "Middle East", "South Asia", "Southeast-East Asia", "Central Asia"]

    # Constructing DataFrame for seaborn plotting
    x_coords = []
    y_coords = []
    corresponding_continent = []

    for continent_idx in range(10):
        for i in range(len(reshaped_reduced_emb[continent_idx])):
            x_coords.append(reshaped_reduced_emb[continent_idx, i, 0])
            y_coords.append(reshaped_reduced_emb[continent_idx, i, 1])
            corresponding_continent.append(continents[continent_idx])
    
    visual_df = pd.DataFrame({ "Reduced Dim 1": x_coords, "Reduced Dim 2": y_coords, "Continent": corresponding_continent })

    # Make seaborn scatter plot
    plt.rcParams.update({'font.size': 18})
    sns.scatterplot(data=visual_df, x="Reduced Dim 1", y="Reduced Dim 2", hue="Continent")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title(plot_title)
    plt.xlim(-250, 250)
    plt.ylim(-250, 250)
    plt.savefig(SAVE_PATH + "/" + save_filename + ".png", bbox_inches='tight')
    plt.clf()
    plt.close()


def plot_spatial_attn_coeff_heatmap(all_attn_coeff_matr, title, save_path):
    continents = ["Africa", "North\nAmerica", "South\nAmerica", "Oceania", "Eastern\nEurope", "Western\nEurope", "Middle\nEast", "South\nAsia", "Southeast-\nEast\nAsia", "Central\nAsia"]
    plt.figure(figsize=(18,12), dpi=100)
    plt.rcParams.update({'font.size': 18})
    sns.heatmap(all_attn_coeff_matr, annot=True, cmap="Blues", fmt=".2f", xticklabels=continents, yticklabels=continents)
    plt.title(title, fontsize=24)
    plt.ylabel("Source Continent")
    plt.xlabel("Destination Continent")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_temporal_attn_coeff_heatmap(all_attn_coeff_matr, seq_len, title, save_path):
    day_idxs = list(range(seq_len))
    # plt.figure(figsize=(12,8), dpi=100)
    plt.figure(figsize=(18,12), dpi=100)
    plt.rcParams.update({'font.size': 18})
    sns.heatmap(all_attn_coeff_matr, annot=True, cmap="Blues", fmt=".2f", xticklabels=day_idxs, yticklabels=day_idxs)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close()


# countries = ["Brazil", "Germany", "Spain", "France", "Britain", "India", "Italy", "Russia", "Turkey", "USA"]
