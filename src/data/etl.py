import pandas as pd

import warnings

warnings.filterwarnings("ignore")


default_mappings = {
    "label_maps": [{1.0: ">= 10 Visits", 0.0: "< 10 Visits"}],
    "protected_attribute_maps": [{1.0: "White", 0.0: "Non-White"}],
}


def default_preprocessing19(df):
    """
    1.Create a new column, RACE that is 'White' if RACEV2X = 1 and HISPANX = 2 i.e. non Hispanic White
      and 'non-White' otherwise
    2. Restrict to Panel 19
    3. RENAME all columns that are PANEL/ROUND SPECIFIC
    4. Drop rows based on certain values of individual features that correspond to missing/unknown - generally < -1
    5. Compute UTILIZATION, binarize it to 0 (< 10) and 1 (>= 10)
    """
    df = df.copy()

    def race(row):
        if (row["HISPANX"] == 2) & (
            row["RACEV2X"] == 1
        ):  # non-Hispanic Whites are marked as WHITE; all others as NON-WHITE#return 'White'
            return "White"
        return "Non-White"

    df["RACEV2X"] = df.apply(lambda row: race(row), axis=1)
    df = df.rename(columns={"RACEV2X": "RACE"})

    df = df[df["PANEL"] == 19]

    # RENAME COLUMNS
    df = df.rename(
        columns={
            "FTSTU53X": "FTSTU",
            "ACTDTY53": "ACTDTY",
            "HONRDC53": "HONRDC",
            "RTHLTH53": "RTHLTH",
            "MNHLTH53": "MNHLTH",
            "CHBRON53": "CHBRON",
            "JTPAIN53": "JTPAIN",
            "PREGNT53": "PREGNT",
            "WLKLIM53": "WLKLIM",
            "ACTLIM53": "ACTLIM",
            "SOCLIM53": "SOCLIM",
            "COGLIM53": "COGLIM",
            "EMPST53": "EMPST",
            "REGION53": "REGION",
            "MARRY53X": "MARRY",
            "AGE53X": "AGE",
            "POVCAT15": "POVCAT",
            "INSCOV15": "INSCOV",
        }
    )

    df = df[df["REGION"] >= 0]  # remove values -1
    df = df[df["AGE"] >= 0]  # remove values -1

    df = df[df["MARRY"] >= 0]  # remove values -1, -7, -8, -9

    df = df[df["ASTHDX"] >= 0]  # remove values -1, -7, -8, -9

    df = df[
        (
            df[
                [
                    "FTSTU",
                    "ACTDTY",
                    "HONRDC",
                    "RTHLTH",
                    "MNHLTH",
                    "HIBPDX",
                    "CHDDX",
                    "ANGIDX",
                    "EDUCYR",
                    "HIDEG",
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
            ]
            >= -1
        ).all(1)
    ]  # for all other categorical features, remove values < -1

    def utilization(row):
        return (
            row["OBTOTV15"]
            + row["OPTOTV15"]
            + row["ERTOT15"]
            + row["IPNGTD15"]
            + row["HHTOTD15"]
        )

    df["TOTEXP15"] = df.apply(lambda row: utilization(row), axis=1)
    lessE = df["TOTEXP15"] < 10.0
    df.loc[lessE, "TOTEXP15"] = 0.0
    moreE = df["TOTEXP15"] >= 10.0
    df.loc[moreE, "TOTEXP15"] = 1.0

    df = df.rename(columns={"TOTEXP15": "UTILIZATION"})
    return df


def default_preprocessing20(df):
    """
    1.Create a new column, RACE that is 'White' if RACEV2X = 1 and HISPANX = 2 i.e. non Hispanic White
      and 'non-White' otherwise
    2. Restrict to Panel 20
    3. RENAME all columns that are PANEL/ROUND SPECIFIC
    4. Drop rows based on certain values of individual features that correspond to missing/unknown - generally < -1
    5. Compute UTILIZATION, binarize it to 0 (< 10) and 1 (>= 10)
    """
    df = df.copy()

    def race(row):
        if (row["HISPANX"] == 2) & (
            row["RACEV2X"] == 1
        ):  # non-Hispanic Whites are marked as WHITE; all others as NON-WHITE
            return "White"
        return "Non-White"

    df["RACEV2X"] = df.apply(lambda row: race(row), axis=1)
    df = df.rename(columns={"RACEV2X": "RACE"})

    #     df = df[df['PANEL'] == 20]

    # RENAME COLUMNS
    df = df.rename(
        columns={
            "FTSTU53X": "FTSTU",
            "ACTDTY53": "ACTDTY",
            "HONRDC53": "HONRDC",
            "RTHLTH53": "RTHLTH",
            "MNHLTH53": "MNHLTH",
            "CHBRON53": "CHBRON",
            "JTPAIN53": "JTPAIN",
            "PREGNT53": "PREGNT",
            "WLKLIM53": "WLKLIM",
            "ACTLIM53": "ACTLIM",
            "SOCLIM53": "SOCLIM",
            "COGLIM53": "COGLIM",
            "EMPST53": "EMPST",
            "REGION53": "REGION",
            "MARRY53X": "MARRY",
            "AGE53X": "AGE",
            "POVCAT15": "POVCAT",
            "INSCOV15": "INSCOV",
        }
    )

    df = df[df["REGION"] >= 0]  # remove values -1
    df = df[df["AGE"] >= 0]  # remove values -1

    df = df[df["MARRY"] >= 0]  # remove values -1, -7, -8, -9

    df = df[df["ASTHDX"] >= 0]  # remove values -1, -7, -8, -9

    df = df[
        (
            df[
                [
                    "FTSTU",
                    "ACTDTY",
                    "HONRDC",
                    "RTHLTH",
                    "MNHLTH",
                    "HIBPDX",
                    "CHDDX",
                    "ANGIDX",
                    "EDUCYR",
                    "HIDEG",
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
            ]
            >= -1
        ).all(1)
    ]  # for all other categorical features, remove values < -1

    def utilization(row):
        return (
            row["OBTOTV15"]
            + row["OPTOTV15"]
            + row["ERTOT15"]
            + row["IPNGTD15"]
            + row["HHTOTD15"]
        )

    df["TOTEXP15"] = df.apply(lambda row: utilization(row), axis=1)
    lessE = df["TOTEXP15"] < 10.0
    df.loc[lessE, "TOTEXP15"] = 0.0
    moreE = df["TOTEXP15"] >= 10.0
    df.loc[moreE, "TOTEXP15"] = 1.0

    df = df.rename(columns={"TOTEXP15": "UTILIZATION"})
    return df


def preprocess(in_fp, out_fp):
    features_to_keep = [
        "REGION",
        "AGE",
        "SEX",
        "RACE",
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
        "PCS42",
        "MCS42",
        "K6SUM42",
        "PHQ242",
        "EMPST",
        "POVCAT",
        "INSCOV",
        "UTILIZATION",
        "PERWT15F",
    ]

    raw_data = pd.read_csv(in_fp + "rawdata.csv")

    df_panel_19 = default_preprocessing19(raw_data)
    df_panel_19_reduced = df_panel_19[features_to_keep]

    df_panel_20 = default_preprocessing20(raw_data)
    df_panel_20_reduced = df_panel_20[features_to_keep]

    df_panel_19_reduced.to_csv(out_fp + "df_panel_19.csv", index=False)
    df_panel_20_reduced.to_csv(out_fp + "df_panel_20.csv", index=False)


def get_data(inpath, outpath):
    preprocess(inpath, outpath)
