# DSC180A-Final-Code-Submission

Learning bias mitigation techniques using the medical expenditure data

# Instructions for obtaining raw data

Follow this link (https://www.kaggle.com/datasets/nanrahman/mepsdata?select=h181.csv) to kaggle.com, download the raw data (h181.csv only) and store it in the /data/raw directory with the filename 'rawdata.csv'.

# Instructions for running scripts

Call 'python run.py test' to run the project using test data, and 'python run.py' to run the project using the raw data once it has been downloaded. 

# run.py

Runs scripts that retrieves dataset, extracts/creates features, trains various models with bias mitigation, and create visualizations based on the results.

## notebooks/

Contains holistic .ipynb that was used to replicate a bias mitigation project and explore our own bias mitigation techniques using an open-source healthcare dataset.

## references/

PDF that describes the features in the MEPS dataset.

## src/

Contains all source code and following directories:

### data/

Contains data preprocessing script that includes renaming features, handling null values, and quantizing continuous features.

### features/

Contains feature creation script that includes converting features to one-hot encodings.

### models/

Contains modeling script that trains various of models (including logistic regression, random forrest) with and without bias mitigation techniques (including reweighing and prejudice remover).

### visualizations/

Creates correlation plots (including race vs insurance and age vs diseases) as well as model drift using the evaluation metrics from models/ between two datasets.

## test/

Contains test target and following directories:

### out/

Output after running data preprocessing and feature creation, as well as results from the model itself:

#### metrics/

Metrics after training various models on the dataset.

#### visualizations/

Plots created after running visualization script.

### testdata/

Test dataset sampled from larger Medical Expenditure Data.

## data/

Contains data target and resulting analyses:

### out/

Output after running data preprocessing and feature creation, as well as results from the model itself:

#### metrics/

Metrics after training various models on the dataset.

#### visualizations/

Plots created after running visualization script.

### raw/

Directory for storage of raw data.


