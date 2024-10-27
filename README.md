![Team Tiny Rainbow Logo](tiny_rainbow_logo.jpeg)

# Description

This repository contains the entry of `team_tiny_rainbow`, from Aeronautics Institute of Technology (ITA), to the Performance Review Commission (PRC) Data Challenge. The main objective in this challenge was to build an **open** Machine Learning (ML) model to accurately infer the Actual TakeOff Weight (ATOW) of a flown flight.

The following artefacts are present in this repository and will be shared publicly:
- The source code used for data processing, feature engineering, and training the predictive models.
- A manuscript (`paper.pdf`) to be submitted to the Journal of Open Aviation Science, an open-access peer-reviewed journal. Moreover, we commit ourselves to publish a public pre-print as soon as the paper is submitted.
- Complementary data used. We have limited ourselves to only information publicly available online. For more information, please refer to the attached manuscript.

# Open Source Commitment

We are committed to go open on the outcome of this challenge!

# Methodology

This section gives an overview of the methodology we used to create our machine learning (ML) models. For more information, we strongly recommend seeing the attached manuscript `paper.pdf`, which is ready to be submitted to the Journal of Open Aviation Science (JOAS).

Our methodology involved preprocessing the data to generate features, which were then fed to supervised ML models. To improve our results, we have also devised the following extra databases (which can be found in the folder `extra_data`):
- OpenFlights Airports Database.
- Global Airport Database.
- Federal Aviation Administration (FAA) Aircraft Database.
- `aircraft_extra_v3.csv`: we created this database of extra aircraft features (i.e. not contained in the FAA database) from manually collecting them from public sources (Wikipedia, technical sheets, factory specifications, and Eurocontrol Data).

The challenge initially provided "flight and trajectory" data for each flight. On top of these initial features, we added features from our complementary data, and engineered new derived features. This process was guided by our experience in Data Science and Aviation and extensive testing. Regarding the ML models, we used the following:
- CatBoost.
- XGBoost.
- LightGBM.
- Neural networks (FastAI's Tabular Learner and SAINT).
- An ensemble of the previous models.

The trajectory data processing was developed in R, while the feature engineering and the ML models were implemented in Python. To communicate between the different modules of the code, we use `.csv` files. In general, our code's pipeline follows these steps:
1. **Trajectory data processing:** process the challenge's data (flight and trajectory data) and generates `trajectories_set.csv` with features for each flight.
2. **Feature engineering**: uses the file generated in the previous step and executes feature engineering by adding complementary data and deriving new features. This generates an augmented dataset composed of `challenge_set_updated.csv` and `submission_set_updated.csv`.
3. **Model training and inference**: uses the augmented dataset to train a particular predictive model and obtain the estimated ATOWs for the submission set.

# How to Run

To execute the code, you initially need to download the datasets from:
https://drive.google.com/drive/folders/1PmEJ880V-0XKuOr-6kcoSqSErOnRBZ88?usp=sharing

We could not keep these files on GitHub due to space limitations.

You should run the scripts/notebooks on the following order:
1. Trajectory Data Processing.
2. Exploratory Data Analysis.
3. Model Training (CatBoost, XGBoost, LightGBM, Neural Networks, or Ensemble).



## R Dependencies

The R code for trajectory data processing was developed and tested under R version 4.4.1 with RStudio version 2024.09.0 build 375. The following dependencies are required:
- `tidyverse`
- `arrow`
- `geosphere`
- `airportr`
- `data.table`
- `REdaS`

These dependencies can be installed with the command `install.packages("dependency_name")`.

## Python Dependencies

The code was developed and tested under Python 3.12. The following Python dependencies are needed:
- `jupyter`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `tqdm`
- `scikit-learn`
- `catboost`
- `xgboost`
- `lightgbm`
- `fastai`
- `optuna`

Execute the following command to install all dependencies:
```
pip3 install jupyter numpy pandas matplotlib seaborn tqdm scikit-learn catboost xgboost lightgbm fastai optuna
```

Furthermore, a `requirements.txt` is also provided.

## Trajectory Data Processing

The R code `get_traj_features4.R` was used for trajectory data processing. This code extracts relevant features from trajectory data for each flight ID in the challenge and (final) submission sets. To run this code, it is necessary to put it in a folder with two subfolders inside: 
- `Challenge Data` with trajectory data parquets;
- `processed` to save the processed files.

The code reads the parquet files inside the `Challenge Data` folder and extracts some features for each flight ID, grouping all entries of each flight ID in a unique entry with all features extracted. Beyond the `.Rda` files generated in the folder `processed` for each day with all flights starting on that day, it also generates the files below in the root folder:
- `trajectory_features4.csv` with all flights from the parquets, by flight ID, with the extracted features;
- `trajectory_features.Rda` R data file only with the flights in the challenge and (final) submission datasets, by flight ID, with the extracted features;
- `trajectory_features.csv` also with only the flights in the challenge and (final) submission datasets, by flight ID, with the extracted features.

Therefore, the file employed in the following steps is `trajectory_features.csv`.

## Exploratory Data Analysis

## CatBoost

Before running any of CatBoost's files, `eda-v20.ipynb` needs to be run in order to generate the feature engineered dataset.

The predictive model based on CatBoost involves the following files:
- `catboost-v20.ipynb`: this is the main notebook for training and inference of the model based on CatBoost. It trains the model on the challenge set and generates the estimated ATOWs on the submission set.
- `catboost-select-v20.ipynb`: uses the method `select_features()` from CatBoost to eliminate features that hurt the model's performance.
- `catboost-optuna.py`: uses Optuna to optimize the CatBoost's hyperparameters.

## XGBoost

## LightGBM

## Neural Networks

## Ensemble

# Team

We are a team of professors and graduate students affiliated with the Aeronautics Institute of Technology (ITA) in Brazil, with expertise in the areas of data science and aviation. The team members are the following:
- Carolina Rutili de Lima.
- Gabriel Adriano de Melo.
- João Basílio Tarelho Szenczuk.
- João Paulo de Andrade Dantas.
- Lucas Orbolato Carvalho.
- Marcos Ricardo Omena de Albuquerque Maximo.
- Mayara Condé Rocha Murça (leader).

# License

This repository is distributed under the GNU GPLv3 license. Please, see `LICENSE.txt` for more information.
