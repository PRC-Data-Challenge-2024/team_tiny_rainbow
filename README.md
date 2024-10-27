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

The trajectory data processing was developed in R, while the feature engineering and the ML models were implemented in Python. To communicate between the different modules of the code, we use `.csv` files.

# How to Run

## R Dependencies

## Python Dependencies

The code was developed and tested under Python 3.12. The following Python dependencies are needed:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `catboost`
- `xgboost`
- `lightgbm`
- `fastai`

Furthermore, for convenience, a `requirements.txt` is provided.

## Exploratory Data Analysis

## CatBoost

## XGBoost

## LightGBM

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