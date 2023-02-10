import pandas as pd
import numpy as np

# model pipelines
from sklearn.pipeline import make_pipeline

# Scalers
from sklearn.preprocessing import StandardScaler

# Feature Selection tools
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# Fairness netrics
from aif360.datasets import BinaryLabelDataset

# categorical feature list given to us by IBM's tutorial for the replication
categorical_features = [
    "REGION",
    "SEX",
    "MARRY",
    "FTSTU",
    "ACTDTY",
    "HONRDC",
    "RTHLTH",
    "MNHLTH",
    "HIBPDX",
    "CHDDX",
    "ANGIDX",
    "MIDX",
    "OHRTDX",
    "STRKDX",
    "EMPHDX",
    "CHBRON",
    "CHOLDX",
    "CANCERDX",
    "DIABDX",
    "JTPAIN",
    "ARTHDX",
    "ARTHTYPE",
    "ASTHDX",
    "ADHDADDX",
    "PREGNT",
    "WLKLIM",
    "ACTLIM",
    "SOCLIM",
    "COGLIM",
    "DFHEAR42",
    "DFSEE42",
    "ADSMOK42",
    "PHQ242",
    "EMPST",
    "POVCAT",
    "INSCOV",
]


def encode_race(df):
    df["RACE"] = (df["RACE"] != "Non-White").astype(int)
    return df


def build_eda_features(df, categorical_feats=categorical_features, seed=57):
    df = encode_race(df)
    one_hot = pd.get_dummies(df, columns=categorical_feats, drop_first=True)
    one_hot = one_hot.sample(frac=1, random_state=seed)
    train, val, test = BinaryLabelDataset(
        df=one_hot,
        label_names=["UTILIZATION"],
        protected_attribute_names=["RACE"],
        privileged_protected_attributes=[0],
    ).split([0.5, 0.8], seed=seed)

    return train, one_hot


def prepare_one_hots(df, return_gs_results=True, to_keep=None, cv=2):
    train, one_hot = build_eda_features(df)

    if return_gs_results:
        eda_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(penalty="l1", solver="saga", max_iter=5000),
        )
        gs = GridSearchCV(
            eda_model,
            {"logisticregression__C": list(np.logspace(-5, 5, 10))},
            n_jobs=-1,
            cv=cv,
            scoring="balanced_accuracy",
        )
        gs.fit(train.features, train.labels.ravel())
        # Utilizing Lasso's ablility to set weights to 0, we use the resulting features for our real models
        keep = (gs.best_estimator_[1].coef_ != 0)[0]
        to_keep = np.array(train.feature_names)[keep]

    one_hot_prime = one_hot[to_keep]
    one_hot_prime["UTILIZATION"] = one_hot["UTILIZATION"]

    if return_gs_results:
        return one_hot_prime, to_keep
    return one_hot_prime


def features(outpath):
    # read df 19 from inpath
    df_19 = pd.read_csv(f"{outpath}/df_panel_19.csv")
    # read df 20 from inpath
    df_20 = pd.read_csv(f"{outpath}/df_panel_20.csv")

    one_hot_df_19, to_keep = prepare_one_hots(df_19)
    one_hot_df_20 = prepare_one_hots(df_20, return_gs_results=False, to_keep=to_keep)

    one_hot_df_19.to_csv(outpath + "feature_prepared_panel_19.csv", index=False)
    one_hot_df_20.to_csv(outpath + "feature_prepared_panel_20.csv", index=False)


def get_features(inpath, outpath):
    features(outpath)
