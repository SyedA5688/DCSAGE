import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_and_return_perturbed_pred_df(model_preds, labels, extended_feedback_preds, perturbed_country_idx, SAVE_PATH, model_idx=None):
    assert len(model_preds) == len(labels) == len(extended_feedback_preds), "Trends have different lengths"
    # countries = ["Brazil", "Germany", "Spain", "France", "Britain", "India", "Italy", "Russia", "Turkey", "USA"]
    continents = ["Africa", "North America", "South America", "Oceania", "Eastern Europe", "Western Europe", "Middle East", "South Asia", "Southeast-East Asia", "Central Asia"]
    country_dataframes = []

    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(30,12))
    fig.suptitle("DCSAGE Predictions When " + continents[perturbed_country_idx] + " Is Perturbed", fontsize= 30)

    idx = 0
    for row in ax:
        for col in row:
            country_preds = []
            country_labels = []
            country_extended_feedback_preds = []
            for i in range(len(model_preds)):
                country_preds.append(float(model_preds[i][idx]))
                country_labels.append(float(labels[i][idx]))
                country_extended_feedback_preds.append(float(extended_feedback_preds[i][idx]))

            visual_df = pd.DataFrame({
                "Regular Predictions": country_preds,
                "Ground Truth": country_labels,
                "Extended Recursive Predictions": country_extended_feedback_preds,
                "Day Index": list(range(len(country_preds))) 
            })
            country_dataframes.append(visual_df)

            sns.lineplot(ax=col, x='Day Index', y='value', hue='variable', data=pd.melt(visual_df, ['Day Index']))
            col.set_title(continents[idx])
            col.set_ylim([0.5, 5.5])
            idx += 1

    filename = "perturbed_pred_trends" if model_idx is None else "unperturb_pred_trends_model_" + str(model_idx)
    plt.savefig(os.path.join(SAVE_PATH, filename + '.png'), bbox_inches='tight')
    plt.clf()
    plt.close()
    return country_dataframes


def plot_and_return_unperturbed_pred_df(model_preds, labels, extended_feedback_preds, SAVE_PATH, model_idx=None):
    assert len(model_preds) == len(labels) == len(extended_feedback_preds), "Trends have different lengths"
    # countries = ["Brazil", "Germany", "Spain", "France", "Britain", "India", "Italy", "Russia", "Turkey", "USA"]
    continents = ["Africa", "North America", "South America", "Oceania", "Eastern Europe", "Western Europe", "Middle East", "South Asia", "Southeast-East Asia", "Central Asia"]
    country_dataframes = []

    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(30,12))
    fig.suptitle("DCSAGE Unperturbed Predictions", fontsize= 30)

    idx = 0
    for row in ax:
        for col in row:
            country_preds = []
            country_labels = []
            country_extended_feedback_preds = []
            for i in range(len(model_preds)):
                country_preds.append(float(model_preds[i][idx]))
                country_labels.append(float(labels[i][idx]))
                country_extended_feedback_preds.append(float(extended_feedback_preds[i][idx]))

            visual_df = pd.DataFrame({
                "Regular Predictions": country_preds,
                "Ground Truth": country_labels,
                "Extended Recursive Predictions": country_extended_feedback_preds,
                "Day Index": list(range(len(country_preds))) 
            })
            country_dataframes.append(visual_df)

            sns.lineplot(ax=col, x='Day Index', y='value', hue='variable', data=pd.melt(visual_df, ['Day Index']))
            col.set_title(continents[idx])
            col.set_ylim([0.5, 5.5])
            idx += 1

    filename = "unperturb_pred_trends" if model_idx is None else "unperturb_pred_trends_model_" + str(model_idx)
    plt.savefig(os.path.join(SAVE_PATH, filename + '.png'), bbox_inches='tight')
    plt.clf()
    plt.close()
    
    return country_dataframes


def get_pred_df(model_preds, labels, extended_feedback_preds):
    """
    This function computes the same dataframe as plot_and_return_perturbed_pred_df(), but this is called if we do
    not need to save figures. Use in r=rolling window analysis, we will not plot everything for every model for
    hundreds of rolling windows.
    """
    assert len(model_preds) == len(labels) == len(extended_feedback_preds), "Trends have different lengths"
    country_dataframes = []

    for idx in range(10):
        country_preds = []
        country_labels = []
        country_extended_feedback_preds = []
        for i in range(len(model_preds)):
            country_preds.append(float(model_preds[i][idx]))
            country_labels.append(float(labels[i][idx]))
            country_extended_feedback_preds.append(float(extended_feedback_preds[i][idx]))

        visual_df = pd.DataFrame({
            "Regular Predictions": country_preds,
            "Ground Truth": country_labels,
            "Extended Recursive Predictions": country_extended_feedback_preds,
            "Day Index": list(range(len(country_preds))) 
        })
        country_dataframes.append(visual_df)
    return country_dataframes
    

def plot_difference_heatmap(sensitivity_scores_lists, SAVE_PATH):
    # countries = ["Brazil", "Germany", "Spain", "France", "Britain", "India", "Italy", "Russia", "Turkey", "USA"]
    continents = ["Africa", "North America", "South America", "Oceania", "Eastern Europe", "Western Europe", "Middle East", "South Asia", "Southeast-East Asia", "Central Asia"]
    heatmap_df = pd.DataFrame(np.array(sensitivity_scores_lists), columns=continents, index=continents)

    plt.figure(figsize=(16,9))
    sns.heatmap(heatmap_df, annot=True)
    plt.title("Heatmap for Difference Matrix")
    plt.ylabel("Perturbed Continent")
    plt.xlabel("Affected Continent")
    plt.savefig(SAVE_PATH + '/difference_heatmap.png', bbox_inches='tight')
    plt.clf()
    plt.close()


def plot_sensitivity_barchart(sensitivity_df, SAVE_PATH):
    plt.figure(figsize=(10, 10), dpi=80)
    sns.barplot(x='Countries', y='Sensitivity', data=sensitivity_df, color="black")  # estimator=np.mean, ci="sd", capsize=0.2, 
    plt.title("Sensitivity Scores")  #  with Standard Deviation Bars
    plt.savefig(SAVE_PATH + '/sensitivity_barchart.png', bbox_inches='tight')  # _with_std
    plt.clf()
    plt.close()


def plot_recursive_and_perturbed_recursive_per_country(perturb_df_nested_lists, regular_df_nested_list, SAVE_PATH):
    """
    perturb_df_nested_lists: (10, 10, 30, 4)
    regular_df_nested_list: (10, 30, 4)
    """
    # countries = ["Brazil", "Germany", "Spain", "France", "Britain", "India", "Italy", "Russia", "Turkey", "USA"]
    continents = ["Africa", "North America", "South America", "Oceania", "Eastern Europe", "Western Europe", "Middle East", "South Asia", "Southeast-East Asia", "Central Asia"]

    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(40, 20))
    fig.suptitle("DCSAGE Recursive vs Perturbed Recursive Predictions", fontsize= 30)

    idx = 0
    for row in ax:
        for col in row:
            # First dataframe has recursive predictions for a country when other 9 countries are perturbed. Perturbed country index first
            visual_df = pd.DataFrame({
                "Brazil Perturbed": perturb_df_nested_lists[0][idx]["Extended Recursive Predictions"],
                "Germany Perturbed": perturb_df_nested_lists[1][idx]["Extended Recursive Predictions"],
                "Spain Perturbed": perturb_df_nested_lists[2][idx]["Extended Recursive Predictions"],
                "France Perturbed": perturb_df_nested_lists[3][idx]["Extended Recursive Predictions"],
                "Britain Perturbed": perturb_df_nested_lists[4][idx]["Extended Recursive Predictions"],
                "India Perturbed": perturb_df_nested_lists[5][idx]["Extended Recursive Predictions"],
                "Italy Perturbed": perturb_df_nested_lists[6][idx]["Extended Recursive Predictions"],
                "Russia Perturbed": perturb_df_nested_lists[7][idx]["Extended Recursive Predictions"],
                "Turkey Perturbed": perturb_df_nested_lists[8][idx]["Extended Recursive Predictions"],
                "USA Perturbed": perturb_df_nested_lists[9][idx]["Extended Recursive Predictions"],
                "Day Index": list(range(len(regular_df_nested_list[idx]["Extended Recursive Predictions"]))) 
            })
            visual_df = visual_df.drop(columns=[continents[idx] + " Perturbed"])

            # Second dataframe has unperturbed recursive predictions for the country
            visual_df2 = pd.DataFrame({
                "Unperturbed Recursive": regular_df_nested_list[idx]["Extended Recursive Predictions"],
                "Day Index": list(range(len(regular_df_nested_list[idx]["Extended Recursive Predictions"]))) 
            })

            sns.lineplot(ax=col, x='Day Index', y='value', hue='variable', data=pd.melt(visual_df, ['Day Index']))
            sns.lineplot(ax=col, x='Day Index', y='value', hue='variable', data=pd.melt(visual_df2, ['Day Index']), linewidth = 4, palette=['black'])
            col.set_title(continents[idx])
            col.set_ylim([1, 5])
            box = col.get_position()
            col.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            col.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            idx += 1

    plt.savefig(os.path.join(SAVE_PATH, '/recursive_vs_perturbed_recursive_2x5.png'), bbox_inches='tight')
    plt.clf()
    plt.close()


def plot_multiple_model_unperturbed_recursive_pred(all_model_unperturb_pred_df_list, args, SAVE_PATH):
    """
    all_model_unperturb_pred_df_list: list of size n for ensemble of n models
    all_model_unperturb_pred_df_list[0]: list of 10 dataframes, one for each country
    all_model_unperturb_pred_df_list[0][0]: First model, Brazil DF containing Extended Recurisve Preds, Regular Preds, and Ground Truth
    """
    # countries = ["Brazil", "Germany", "Spain", "France", "Britain", "India", "Italy", "Russia", "Turkey", "USA"]
    continents = ["Africa", "North America", "South America", "Oceania", "Eastern Europe", "Western Europe", "Middle East", "South Asia", "Southeast-East Asia", "Central Asia"]
    recursive_pred_len = args["recursive_pred_len"]
    num_models = args["num_models"]

    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(30,12))
    fig.suptitle("DCSAGE " + str(num_models) + " Model Unperturbed Recursive Predictions", fontsize= 30)

    country_idx = 0
    for row in ax:
        for col in row:
            country_subplot_dict = { "Model_" + str(model_idx): all_model_unperturb_pred_df_list[model_idx][country_idx]["Extended Recursive Predictions"] for model_idx in range(num_models) }
            country_subplot_dict["Day Index"] = list(range(recursive_pred_len))
            visual_df = pd.DataFrame(country_subplot_dict)

            sns.lineplot(ax=col, x='Day Index', y='value', hue='variable', data=pd.melt(visual_df, ['Day Index']))
            col.set_title(continents[country_idx])
            col.legend(ncol=2)
            col.set_ylim([0.5, 5.5])
            country_idx += 1

    filename = str(num_models) + "_models_rec_pred"
    plt.savefig(os.path.join(SAVE_PATH, filename + '.png'), bbox_inches='tight')
    plt.clf()
    plt.close()


#########################
# Sensitivity Score Plots
#########################
def plot_multiple_model_roll_win_sensitivity_scores_lineplot(roll_win_summed_differences_nested_list, args, SAVE_PATH):
    """
    roll_win_sensitivity_score_nested_list is (34, 10, 10, 10)  - 34 roll windows, 10 models, then 10x10 summed differences
    """
    # countries = ["Brazil", "Germany", "Spain", "France", "Britain", "India", "Italy", "Russia", "Turkey", "USA"]
    continents = ["Africa", "North America", "South America", "Oceania", "Eastern Europe", "Western Europe", "Middle East", "South Asia", "Southeast-East Asia", "Central Asia"]
    num_models = args["num_models"]

    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(30,15))
    fig.suptitle("DCSAGE " + str(num_models) + " Model Rolling Window Sensitivity Scores", fontsize= 30)

    country_idx = 0
    for row in ax:
        for col in row:
            all_model_country_scores = []
            for model_idx in range(num_models):
                model_sensitivity_scores = []
                for roll_win_idx in range(len(roll_win_summed_differences_nested_list)):
                    sensitivity = np.nansum(np.array(roll_win_summed_differences_nested_list[roll_win_idx][model_idx][country_idx]))
                    model_sensitivity_scores.append(sensitivity)
                all_model_country_scores.append(model_sensitivity_scores)
            
            # all_model_country_scores is (num_models, num_roll_windows)
            country_subplot_dict = { "Model_" + str(model_idx): all_model_country_scores[model_idx] for model_idx in range(num_models) }
            country_subplot_dict["Rolling Window Index"] = list(range(len(roll_win_summed_differences_nested_list)))
            visual_df = pd.DataFrame(country_subplot_dict)

            sns.lineplot(ax=col, x='Rolling Window Index', y='Sensitivity Scores', hue='Model', data=pd.melt(visual_df, ['Rolling Window Index'], value_name="Sensitivity Scores", var_name="Model"))
            col.set_title(continents[country_idx])
            col.legend(ncol=2)
            col.set_ylim([-5, 5])
            country_idx += 1

    filename = str(num_models) + "_models_roll_win_sens_scores"
    plt.savefig(os.path.join(SAVE_PATH, filename + '.png'), bbox_inches='tight', facecolor='white')
    plt.clf()
    plt.close()


def plot_multiple_model_roll_win_sens_distribution(roll_win_aggreg_diff_nested_list, args, SAVE_PATH):
    """
    roll_win_aggreg_diff_nested_list is (34, 10, 10, 10)  - 34 roll windows, 10 models, then 10x10 summed differences
    """
    # countries = ["Brazil", "Germany", "Spain", "France", "Britain", "India", "Italy", "Russia", "Turkey", "USA"]
    continents = ["Africa", "North America", "South America", "Oceania", "Eastern Europe", "Western Europe", "Middle East", "South Asia", "Southeast-East Asia", "Central Asia"]
    num_models = args["num_models"]

    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(35,12))
    fig.suptitle("DCSAGE " + str(num_models) + " Model Rolling Window Sensitivity Score Distributions", fontsize= 30)

    pert_country_idx = 0
    for row in ax:
        for col in row:
            country_sensitivity_scores = []
            for model_idx in range(num_models):
                for roll_win_idx in range(len(roll_win_aggreg_diff_nested_list)):
                    sensitivity = np.nansum(np.array(roll_win_aggreg_diff_nested_list[roll_win_idx][model_idx][pert_country_idx]))
                    country_sensitivity_scores.append(sensitivity)
            
            visual_df = pd.DataFrame({
                "Sensitivity Score": country_sensitivity_scores,
            })

            sns.histplot(ax=col, x='Sensitivity Score', data=visual_df, kde=True)
            mean = np.array(country_sensitivity_scores).mean()
            median = np.median(np.array(country_sensitivity_scores))
            stddev = np.array(country_sensitivity_scores).std()
            col.set_title(continents[pert_country_idx] + ", Mean " + str(round(mean, 2)) + ", Median: " + str(round(median, 2)) + ", Std: " + str(round(stddev, 2)))
            col.set_xlim([-30, 30])
            pert_country_idx += 1

    filename = str(num_models) + "_models_sens_score_distrib"
    plt.savefig(os.path.join(SAVE_PATH, filename + '.png'), bbox_inches='tight', facecolor="white")
    plt.clf()
    plt.close()


#########################
# Summed Difference Plots
#########################
def plot_multiple_model_roll_win_summed_differences_lineplot(roll_win_summed_differences_nested_list, args, SAVE_PATH):
    """
    roll_win_sensitivity_score_nested_list is (34, 10, 10, 10)  - 34 roll windows, 10 models, then 10x10 summed differences
    """
    # countries = ["Brazil", "Germany", "Spain", "France", "Britain", "India", "Italy", "Russia", "Turkey", "USA"]
    continents = ["Africa", "North America", "South America", "Oceania", "Eastern Europe", "Western Europe", "Middle East", "South Asia", "Southeast-East Asia", "Central Asia"]
    num_models = args["num_models"]

    for perturbed_country_idx in range(10):
        fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(30,15))
        fig.suptitle("DCSAGE " + str(num_models) + " Model Rolling Window Summed Differences When " + continents[perturbed_country_idx] + " is Perturbed", fontsize= 30)

        country_idx = 0
        for row in ax:
            for col in row:
                all_model_country_scores = []
                for model_idx in range(num_models):
                    country_scores = []
                    for roll_win_idx in range(len(roll_win_summed_differences_nested_list)):
                        country_scores.append(roll_win_summed_differences_nested_list[roll_win_idx][model_idx][perturbed_country_idx][country_idx])
                    all_model_country_scores.append(country_scores)
                
                # all_model_country_scores is (num_models, num_roll_windows)
                country_subplot_dict = { "Model_" + str(model_idx): all_model_country_scores[model_idx] for model_idx in range(num_models) }
                country_subplot_dict["Rolling Window Index"] = list(range(len(roll_win_summed_differences_nested_list)))
                visual_df = pd.DataFrame(country_subplot_dict)

                if country_idx != perturbed_country_idx:
                    sns.lineplot(ax=col, x='Rolling Window Index', y='Summed Difference', hue='Model', data=pd.melt(visual_df, ['Rolling Window Index'], value_name="Summed Difference", var_name="Model"))
                    col.set_title(continents[country_idx])
                    col.legend(ncol=2)
                    col.set_ylim([-10, 10])
                country_idx += 1

        filename = str(num_models) + "_models_" + continents[perturbed_country_idx] + "_pert_summed_diffs"
        plt.savefig(os.path.join(SAVE_PATH, filename + '.png'), bbox_inches='tight')
        plt.clf()
    plt.close()


def plot_multiple_model_roll_win_summed_diff_distribution(roll_win_sensitivity_score_nested_list, args, SAVE_PATH):
    """
    roll_win_aggreg_diff_nested_list is (34, 10, 10, 10)  - 34 roll windows, 10 models, then 10x10 summed differences
    """
    # countries = ["Brazil", "Germany", "Spain", "France", "Britain", "India", "Italy", "Russia", "Turkey", "USA"]
    continents = ["Africa", "North America", "South America", "Oceania", "Eastern Europe", "Western Europe", "Middle East", "South Asia", "Southeast-East Asia", "Central Asia"]
    num_models = args["num_models"]

    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(35,12))
    fig.suptitle("DCSAGE " + str(num_models) + " Model Rolling Window Summed Difference Distributions", fontsize= 30)

    country_idx = 0
    for row in ax:
        for col in row:
            summed_diffs = []
            for model_idx in range(num_models):
                for roll_win_idx in range(len(roll_win_sensitivity_score_nested_list)):
                    for perturbed_country_idx in range(10):
                        if perturbed_country_idx != country_idx:
                            summed_diffs.append(roll_win_sensitivity_score_nested_list[roll_win_idx][model_idx][perturbed_country_idx][country_idx])
            
            print(continents[country_idx], "has", len(summed_diffs), "summed differences in distribution.")
            visual_df = pd.DataFrame({
                "Summed Difference": summed_diffs,
            })

            sns.histplot(ax=col, x='Summed Difference', data=visual_df, kde=True)
            mean = np.array(summed_diffs).mean()
            stddev = np.array(summed_diffs).std()
            col.set_title(continents[country_idx] + ", Mean " + str(round(mean, 2)) + ", Std: " + str(round(stddev, 2)))
            col.set_xlim([-6, 6])
            country_idx += 1

    filename = str(num_models) + "_models_summed_diff_distrib"
    plt.savefig(os.path.join(SAVE_PATH, filename + '.png'), bbox_inches='tight', facecolor="white")
    plt.clf()
    plt.close()


def plot_tsne_reduced_embeddings(reduced_embeddings_matrix, plot_title, save_filename, SAVE_PATH):
    pass
