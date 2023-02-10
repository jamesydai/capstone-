import sys
sys.path.insert(0, '../')

import pandas as pd
import numpy as np

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.sklearn import metrics

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover
from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score

def best_threshold(fitted_model,df,groups,best_thresh=None,model_type="sklearn",scaler=None):
    if model_type == "sklearn":
        y_val_pred_prob = fitted_model.predict_proba(df.features)
        pos_ind = np.where(fitted_model.classes_ == df.favorable_label)[0][0]
    elif model_type == "aif360":
        new_df = df.copy()
        new_df.features = scaler.transform(new_df.features)
        y_val_pred_prob = fitted_model.predict(new_df).scores
        pos_ind = 0
    if best_thresh == None:
        best_bar = -1
        best_thresh = -1
        for thresh in np.linspace(0.01, 0.5, 50):
            y_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)
            y_true = df.labels.ravel()
            df_pred = df.copy()
            df_pred.labels = y_pred.reshape(-1,1)
            metric = ClassificationMetric(df, 
                                          df_pred, 
                                          unprivileged_groups=groups[0], 
                                          privileged_groups=groups[1])
            curr_bar = (metric.true_positive_rate() + metric.true_negative_rate()) / 2
            if curr_bar > best_bar:
                best_bar = curr_bar
                best_thresh = thresh
    y_val_pred = (y_val_pred_prob[:, pos_ind] > best_thresh).astype(np.float64)
    y_true = df.labels.ravel()
    df_pred = df.copy()
    df_pred.scores = y_val_pred_prob[:, pos_ind].reshape(-1,1)
    df_pred.labels = y_val_pred.reshape(-1,1)
    return (df_pred, best_thresh)

def print_stats(df,df_pred,groups, model_type):
    metric = ClassificationMetric(df, 
                                  df_pred, 
                                  unprivileged_groups=groups[0], 
                                  privileged_groups=groups[1])
    metric_dict = {}
    metric_dict['bal_acc']= (metric.true_positive_rate() + metric.true_negative_rate()) / 2
    metric_dict['avg_odds_diff'] = metric.average_odds_difference()
    metric_dict['disp_imp'] = metric.disparate_impact()
    metric_dict['stat_par_diff'] = metric.statistical_parity_difference()
    metric_dict['eq_opp_diff'] = metric.equal_opportunity_difference()
    metric_dict['theil_ind'] = metric.theil_index()
    metric_dict['model'] = model_type
    disp_imp_at_best_ind = 1 - min(metric_dict['disp_imp'], 1/metric_dict['disp_imp'])
    return metric_dict

def create_df(dct_lst):
    new_dct = {}
    for dct in dct_lst:
        for key in dct_lst[0]:
            if key in new_dct:
                new_dct[key] += [dct[key]]
            else:
                new_dct[key] = [dct[key]]
    return pd.DataFrame(new_dct).set_index('model').T

def perform_modeling(inpath, outpath):
	df_19 = pd.read_csv(outpath+'feature_prepared_panel_19.csv')
	df_20 = pd.read_csv(outpath+'feature_prepared_panel_20.csv')

	train, val, test = BinaryLabelDataset(df=df_19,
	                                      label_names=["UTILIZATION"],
	                                      protected_attribute_names=["RACE"], 
	                                      privileged_protected_attributes = [0]).split([0.5, 0.8])

	model_drift_test = BinaryLabelDataset(df=df_20,
	                                      label_names=["UTILIZATION"],
	                                      protected_attribute_names=["RACE"], 
	                                      privileged_protected_attributes = [0])

	sens_ind = 0
	sens_attr = train.protected_attribute_names[sens_ind]

	unprivileged_groups = [{sens_attr: v} for v in
	                       train.unprivileged_protected_attributes[sens_ind]]
	privileged_groups = [{sens_attr: v} for v in
	                     train.privileged_protected_attributes[sens_ind]]

	GROUPS = (unprivileged_groups,privileged_groups)

	# Train logistic regression and collect  stats
	lr_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
	grid_lr_model = GridSearchCV(lr_model, 
	                  {'logisticregression__C': list(np.logspace(-5, 5, 10))}, 
	                  n_jobs = -1, 
	                  cv = 2)
	grid_lr_model.fit(train.features, train.labels.ravel())
	val_pred_lr, best_thresh_lr = best_threshold(grid_lr_model,val,GROUPS)
	lr_val_metric_arrs = print_stats(val, val_pred_lr, GROUPS, 'Validation Base Logistic Regression')
	test_pred_lr, best_thresh_lr = best_threshold(grid_lr_model,test,GROUPS,best_thresh_lr)
	lr_test_metric_arrs = print_stats(test, test_pred_lr, GROUPS, 'Base Logistic Regression')
	test_pred_lr_20, best_thresh_lr = best_threshold(grid_lr_model,model_drift_test,GROUPS,best_thresh_lr)
	lr_test_metric_arrs_20 = print_stats(model_drift_test, test_pred_lr_20, GROUPS, '2020 Base Logistic Regression')

	# Train random forest and collect stats
	rf_model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=90, min_samples_leaf=18))
	rf_model.fit(train.features, train.labels.ravel())
	val_pred_rf, best_thresh_rf = best_threshold(rf_model,val,GROUPS)
	rf_val_metric_arrs = print_stats(val, val_pred_rf, GROUPS, 'Validation Base Random Forest')
	test_pred_rf, best_thresh_rf = best_threshold(rf_model,test,GROUPS,best_thresh_rf)
	rf_test_metric_arrs = print_stats(test, test_pred_rf, GROUPS, 'Base Random Forest')
	test_pred_rf_20, best_thresh_rf = best_threshold(rf_model,model_drift_test,GROUPS,best_thresh_rf)
	rf_test_metric_arrs_20 = print_stats(model_drift_test, test_pred_rf_20, GROUPS, '2020 Base Random Forest')

	# Reweight data
	RW = Reweighing(unprivileged_groups=unprivileged_groups,
	                privileged_groups=privileged_groups)
	RW_train = RW.fit_transform(train)

	# Train logistic regression on reweighted data and collect stats
	lr_model = make_pipeline(StandardScaler(), LogisticRegression(solver='saga',max_iter=5000))
	RW_grid_lr_model = GridSearchCV(lr_model, 
	                  {'logisticregression__C': list(np.logspace(-5, 5, 10))}, 
	                  n_jobs = -1, 
	                  cv = 2)
	RW_grid_lr_model.fit(RW_train.features, RW_train.labels.ravel())
	RW_val_pred_lr, RW_best_thresh_lr = best_threshold(RW_grid_lr_model,val,GROUPS)
	RW_lr_val_metric_arrs = print_stats(val, RW_val_pred_lr, GROUPS, 'Validation Reweighted Logistic Regression')
	RW_test_pred_lr, RW_best_thresh_lr = best_threshold(RW_grid_lr_model,test,GROUPS,RW_best_thresh_lr)
	RW_lr_test_metric_arrs = print_stats(test, RW_test_pred_lr, GROUPS, 'Reweighted Logistic Regression')
	RW_test_pred_lr_20, RW_best_thresh_lr = best_threshold(RW_grid_lr_model,model_drift_test,GROUPS,RW_best_thresh_lr)
	RW_lr_test_metric_arrs_20 = print_stats(model_drift_test, RW_test_pred_lr_20, GROUPS, '2020 Reweighted Logistic Regression')

	# Train random forest on reweighted data and collect stats
	RW_rf_model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=90, min_samples_leaf=18))
	RW_rf_model.fit(RW_train.features, RW_train.labels.ravel())
	RW_val_pred_rf, RW_best_thresh_rf = best_threshold(RW_rf_model,val,GROUPS)
	RW_rf_val_metric_arrs = print_stats(val, RW_val_pred_rf, GROUPS, 'Validation Reweighted Random Forest')
	RW_test_pred_rf, RW_best_thresh_rf = best_threshold(RW_rf_model,test,GROUPS,RW_best_thresh_rf)
	RW_rf_test_metric_arrs = print_stats(test, RW_test_pred_rf, GROUPS, 'Reweighted Random Forest')
	RW_test_pred_rf_20, RW_best_thresh_rf = best_threshold(RW_rf_model,model_drift_test,GROUPS,RW_best_thresh_rf)
	RW_rf_test_metric_arrs_20 = print_stats(model_drift_test, RW_test_pred_rf_20, GROUPS, '2020 Reweighted Random Forest')

	# Create prejudice remover and collect stats
	pr_model = PrejudiceRemover(sensitive_attr=sens_attr, eta=25.0)
	scaler = StandardScaler()
	new_train = train.copy()
	new_train.features = scaler.fit_transform(new_train.features)
	pr_model = pr_model.fit(new_train)
	val_pred_pr, best_thresh_pr = best_threshold(pr_model,val,GROUPS,model_type="aif360",scaler=scaler)
	pr_val_metric_arrs = print_stats(val, val_pred_pr, GROUPS, 'Validation Prejudice Remover')
	test_pred_pr, best_thresh_pr = best_threshold(pr_model,test,GROUPS,
	                                              best_thresh=best_thresh_pr,
	                                              model_type="aif360",
	                                              scaler=scaler)
	pr_test_metric_arrs = print_stats(test, test_pred_pr, GROUPS, 'Prejudice Remover')
	test_pred_pr_20, best_thresh_pr = best_threshold(pr_model,model_drift_test,GROUPS,
	                                              best_thresh=best_thresh_pr,
	                                              model_type="aif360",
	                                              scaler=scaler)
	pr_test_metric_arrs_20 = print_stats(model_drift_test, test_pred_pr_20, GROUPS, '2020 Prejudice Remover')

	#Create Reject Options Classifier
	ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups,
	                privileged_groups=privileged_groups)

	# Fit ROC and collect stats on Logistic Regression
	ROC_lr = ROC.fit(val,val_pred_lr)
	ROC_val_lr = ROC.predict(val_pred_lr)
	roc_lr_val_metric_arrs = print_stats(val, ROC_val_lr, GROUPS, 'Validation Reject Options Classifier + Logistic Regression')
	ROC_test_lr = ROC.predict(test_pred_lr)
	roc_lr_test_metric_arrs = print_stats(test, ROC_test_lr, GROUPS, 'Reject Options Classifier + Logistic Regression')
	ROC_test_lr_20 = ROC.predict(test_pred_lr_20)
	roc_lr_test_metric_arrs_20 = print_stats(model_drift_test, ROC_test_lr_20, GROUPS, '2020 Reject Options Classifier + Logistic Regression')

	# Fit ROC and collect stats on Random Forest
	ROC_rf = ROC.fit(val,val_pred_rf)
	ROC_val_rf = ROC.predict(val_pred_rf)
	roc_rf_val_metric_arrs = print_stats(val, ROC_val_rf, GROUPS, 'Validation Reject Options Classifier + Random Forest')
	ROC_test_rf = ROC.predict(test_pred_rf)
	roc_rf_test_metric_arrs = print_stats(test, ROC_test_rf, GROUPS, 'Reject Options Classifier + Random Forest')
	ROC_test_rf_20 = ROC.predict(test_pred_rf_20)
	roc_rf_test_metric_arrs_20 = print_stats(model_drift_test, ROC_test_rf_20, GROUPS, '2020 Reject Options Classifier + Random Forest')

	# Make lists to prep output dataframe
	lst1 = ([lr_val_metric_arrs, rf_val_metric_arrs, 
	         RW_lr_val_metric_arrs, RW_rf_val_metric_arrs, 
	         pr_val_metric_arrs, roc_lr_val_metric_arrs, roc_rf_val_metric_arrs])
	lst2 = ([lr_test_metric_arrs, rf_test_metric_arrs, 
	         RW_lr_test_metric_arrs, RW_rf_test_metric_arrs, 
	         pr_test_metric_arrs, roc_lr_test_metric_arrs, roc_rf_test_metric_arrs])
	lst3 = ([lr_test_metric_arrs_20, rf_test_metric_arrs_20, 
	         RW_lr_test_metric_arrs_20, RW_rf_test_metric_arrs_20, 
	         pr_test_metric_arrs_20, roc_lr_test_metric_arrs_20, roc_rf_test_metric_arrs_20])

	create_df(lst1).to_csv(outpath+"metrics/validation_metrics.csv")
	create_df(lst2).to_csv(outpath+"metrics/test_metrics_19.csv")
	create_df(lst3).to_csv(outpath+"metrics/test_metrics_20.csv")