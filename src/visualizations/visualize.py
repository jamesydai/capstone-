from venv import create
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from IPython.display import Markdown, display
import seaborn as sns


def plot_metrics(
        image_name: str, title: str, results_list: list,
        result_type_list: list,
        hue_title: str = "Model Type",
        colors: str = ["#bdbdbd", "#99d8c9", "#bebada"]):
    """
    Plots the metrics.

    Parameters
    -----------
    results_list: list of dictionaries
        Should be a list of dictionaries, where each dictionary is the model's results (e.g. lr_test_metric_arrs)

    result_type_list: list of strings
        Should be a list of strings, where each string is the label indicating what result list it represents
        (e.g. ['No Bias Mitigation', 'Reweighing'])

    hue_title: str
        Title for the 'hue' of the plot legend and grouping. (e.g. 'Model Type')

    colors: list
        Plot palette colors.


    Outputs
    -------
    Bar plot.

    Returns
    --------
    The data used to generate the plot.
    """

    model_results = pd.concat([pd.Series(results)
                              for results in results_list], axis=0)
    model_results.name = "Score"
    model_type = np.concatenate(
        [[indicator] * 6 for indicator in result_type_list])
    model_results = model_results.to_frame().reset_index().rename(
        columns={"index": "Metric"})
    model_results[hue_title] = model_type

    # renaming stuff for plot clarity
    metrics = {"bal_acc": "Balanced Accuracy",
               "avg_odds_diff": "Average Odds Diff.",
               "disp_imp": "Disparate Impact",
               "stat_par_diff": "Statistical Parity Diff.",
               "eq_opp_diff": "Equal Opportunity Diff.",
               "theil_ind": "Theil Index"}
    model_results = model_results.replace(metrics)

    sns.set_palette(sns.color_palette(colors))
    plot = sns.barplot(data=model_results,
                       x="Metric",
                       y="Score",
                       hue=hue_title,
                       saturation=1.25)

    plot.axhline(0, color="black", linewidth=0.6)
    plot.set_xlabel("Metric", fontsize=14)
    plot.set_ylabel("Score", fontsize=14)
    plt.xticks(rotation=30)
    plt.title(title)
    plt.savefig(image_name)
    plt.close()
    return model_results


def create_visualizations(inpath, outpath):
    # read df 19 from inpath
    df_19 = pd.read_csv(f"{outpath}df_panel_19.csv")
    # read df 20 from inpath
    df_20 = pd.read_csv(f"{outpath}df_panel_20.csv")
    

    race = df_19['RACE']

    non_categorical = ['AGE', 'K6SUM42', 'MCS42', 'PCS42', 'PERWT15F']

    # Non-Whites with Health Insurance
    df_19[race == 'Non-White']['INSCOV'].value_counts().rename({1: 'Private', 2: 'Public', 3: 'Uninsured'}) .plot(
        kind='barh', title='Non-White Individuals Health Insurance Status')
    plt.savefig(f'{outpath}visualizations/non_white_health_ins.jpg')
    plt.close()

    # White with Health Insurance
    df_19[race != 'Non-White']['INSCOV'].value_counts().rename({1: 'Private', 2: 'Public', 3: 'Uninsured'}).plot(
        kind='barh', title='White Individuals Health Insurance Status')
    plt.savefig(f'{outpath}visualizations/white_health_ins.jpg')
    plt.close()

    # High Cholesterol Pressure by Age
    binary_df_19 = df_19.copy()
    
    binary_df_19.loc[binary_df_19['CHOLDX'] != 1, 'CHOLDX'] = 0
    binary_df_19.groupby('AGE')['CHOLDX'].mean().plot(title='Coronary Heart Disease by Age')
    plt.savefig(f'{outpath}visualizations/cholesterol_by_age.jpg')
    plt.close()

    # Coronary Heart Disease by Age
    binary_df_19 = df_19.copy()
    
    binary_df_19.loc[binary_df_19['CHDDX'] != 1, 'CHDDX'] = 0
    binary_df_19.groupby('AGE')['CHDDX'].mean().plot(title='Coronary Heart Disease by Age')
    plt.savefig(f'{outpath}visualizations/heart_disease_by_age.jpg')
    plt.close()

    # Correlation plot 1
    corr_plot_19 = pd.plotting.scatter_matrix(
        df_19[np.append(non_categorical, 'UTILIZATION')])
    plt.savefig(f'{outpath}visualizations/correlation_19.jpg')
    plt.close()

    # Correlation plot 2
    corr_plot_20 = pd.plotting.scatter_matrix(
        df_20[np.append(non_categorical, 'UTILIZATION')])
    plt.savefig(f'{outpath}visualizations/correlation_20.jpg')
    plt.close()

    # Model Drift
    metrics_19 = f"{outpath}metrics/test_metrics_19.csv"
    metrics_20 = f"{outpath}metrics/test_metrics_20.csv"
    _19 = pd.read_csv(metrics_19)
    _20 = pd.read_csv(metrics_20)

    lr_test_metric_arrs = _19['Base Logistic Regression']
    lr_test_metric_arrs_20 = _20['2020 Base Logistic Regression']
    # log model drift plots
    log_model_drift_res = plot_metrics(f'{outpath}visualizations/drift1.jpg',
                                       'Logistic Regression Drift',
                                       [lr_test_metric_arrs, lr_test_metric_arrs_20], [
                                           "Panel 19", "Panel 20"], hue_title="Panel")

    RW_lr_test_metric_arrs = _19['Reweighted Logistic Regression']
    RW_lr_test_metric_arrs_20 = _20['2020 Reweighted Logistic Regression']
    # reweighed log model drift_plots
    rw_log_model_drift_res = plot_metrics(f'{outpath}visualizations/drift2.jpg',
                                          'Reweighted Logistic Regression Drift',
                                          [RW_lr_test_metric_arrs, RW_lr_test_metric_arrs_20], [
                                              "Panel 19", "Panel 20"], hue_title="Panel")

    rf_test_metric_arrs = _19['Base Random Forest']
    rf_test_metric_arrs_20 = _20['2020 Base Random Forest']
    # random forest drift plots
    rf_model_drift_res = plot_metrics(f'{outpath}visualizations/drift3.jpg',
                                      'Random Forest Drift',
                                      [rf_test_metric_arrs, rf_test_metric_arrs_20], [
                                          "Panel 19", "Panel 20"], hue_title="Panel")

    RW_rf_test_metric_arrs = _19['Reweighted Random Forest']
    RW_rf_test_metric_arrs_20 = _20['2020 Reweighted Random Forest']
    # reweighed random forest drift plots
    rw_rf_model_drift_res = plot_metrics(f'{outpath}visualizations/drift4.jpg',
                                         'Reweighted Random Forest Drift',
                                         [RW_rf_test_metric_arrs, RW_rf_test_metric_arrs_20], [
                                             "Panel 19", "Panel 20"], hue_title="Panel")

    pr_test_metric_arrs = _19['Prejudice Remover']
    pr_test_metric_arrs_20 = _20['2020 Prejudice Remover']
    # prejudice remover drift plots
    prejudice_remover_drift_res = plot_metrics(f'{outpath}visualizations/drift5.jpg',
                                               'Prejudice Remover Drift',
                                               [pr_test_metric_arrs, pr_test_metric_arrs_20], [
                                                   "Panel 19", "Panel 20"], hue_title="Panel")

    roc_rf_test_metric_arrs = _19['Reject Options Classifier + Random Forest']
    roc_rf_test_metric_arrs_20 = _20['2020 Reject Options Classifier + Random Forest']
    reject_options_classifier_rf__drift_res = plot_metrics(f'{outpath}visualizations/drift6.jpg',
                                                           'Reject Options Classifier + Random Forest Drift',
                                                           [roc_rf_test_metric_arrs, roc_rf_test_metric_arrs_20], ["Panel 19", "Panel 20"], hue_title="Panel")
