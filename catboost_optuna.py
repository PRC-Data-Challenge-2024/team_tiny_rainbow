from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool, metrics, cv
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import root_mean_squared_error
import optuna

# Load the dataset after the exploratory data analysis
# challenge_set_updated = pd.read_csv("./data/challenge_set_updated_v7.csv")
challenge_set_updated = pd.read_csv("./data/challenge_set_updated_v9_median.csv")
submission_set = pd.read_csv("./data/submission_set.csv")
# submission_set_updated = pd.read_csv("./data/submission_set_updated_v7.csv")
submission_set_updated = pd.read_csv("./data/submission_set_updated_v9_median.csv")

# If necessary change this part to test the model before the training process
df = challenge_set_updated.iloc[:,:]
# df = challenge_set_updated.sample(frac=0.001)

# Separating features and target variable
X = df.drop('tow', axis=1)
y = df['tow']

df.head()

to_drop = ['offblock_to_arrival_duration', 'normalized_taxi_ratio', 'MALW_kg', 'wind_distance_ARR_100', 'average_airspeed_ARR_100', 'track_variation_ARR_100', 'is_offblock_weekend', 'Num_Engines', 'flown_distance_ARR_100', 'average_humidity_ARR_100', 'average_temperature_ARR_100', 'wind_distance_DEP_100', 'arrival_minute', 'track_variation_ENR', 'groundspeed_ARR_100', 'average_vertical_rate_ARR_100', 'taxiout_time', 'track_variation_DEP_100', 'average_airspeed_DEP_100', 'offblock_minute', 'average_airspeed_ENR', 'specific_energy_ENR',
'taxi_ratio', 'average_humidity_DEP_100', 'specific_energy_ARR_100', 'is_offblock_rush_hour', 'wind_distance_ENR', 'groundspeed_ENR', 'altitude_difference', 'average_vertical_rate_ENR', 'bearing', 'Altitude_ades']

cat_names = ['adep',
             'ades',
             'aircraft_type', 
             'wtc', 
             'airline',
             'offblock_hour',
             'offblock_minute', 
             'offblock_day_of_week',
             'offblock_month',
             'offblock_week_of_year', 
             'offblock_season', 
             'arrival_hour',
             'arrival_minute',
             'is_offblock_weekend',
             'is_offblock_rush_hour',
             'flight_duration_category',                       
             'adep_region', 
             'ades_region', 
             'same_country_flight',
             'same_region_flight',                        
             'flight_direction',
             'is_intercontinental',
             'Manufacturer',
             'Model_FAA',
             'Physical_Class_Engine',
             'FAA_Weight',
             'adep_geo_cluster',
             'ades_geo_cluster']

selected_cat_names = [x for x in cat_names if x not in to_drop]
             
X = df.drop('tow', axis=1)
y = df.tow

X.drop(to_drop, axis=1, inplace=True)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_pool = Pool(X_train, y_train, cat_features=selected_cat_names)
val_pool = Pool(X_val, y_val, cat_features=selected_cat_names)

def objective(trial):
    # Taken from: 
    # https://deepnote.com/app/svpino/Tuning-Hyperparameters-with-Optuna-ea1a123d-8d2f-4e20-8f22-95f07470d557
    # https://forecastegy.com/posts/catboost-hyperparameter-tuning-guide-with-optuna/
    params = {
        #'learning_rate' : trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'learning_rate' : 0.1,
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 100, log=True),
        'subsample': trial.suggest_float('subsample', 0.05, 1),
        'random_strength': trial.suggest_float('random_strength', 10, 50),
        'depth': trial.suggest_int('depth', 1, 15),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.05, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30),
        'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 15),
    }
    
    model = CatBoostRegressor(
        iterations=5000,
        eval_metric=metrics.RMSE(),
        random_seed=42,
        verbose=False,
        objective=metrics.RMSE(),
        task_type='CPU', # training on CPU because my GPU doesn't have enough memory
        use_best_model=True,
        od_type='Iter',
        od_wait=50,
        **params,
    )


    model.fit(train_pool, eval_set=val_pool)
    best_rmse = model.get_best_score()['validation']['RMSE']
    #cv_data = cv(
    #    train_pool,
    #    model.get_params(),
    #    logging_level='Silent',
    #)
    #best_rmse = np.min(cv_data['test-RMSE-mean'])

    return best_rmse
    
study = optuna.create_study(study_name='catboost_tunning_v9', storage='sqlite:///catboost.db', 
			    direction='minimize', load_if_exists=True)
study.optimize(objective, n_trials=30)

# Display the best hyperparameters found
print(f"Best trial: {study.best_trial.params}")

# Train the final model with the best parameters
best_params = study.best_trial.params
best_model = CatBoostRegressor(
    iterations=10000,
    eval_metric=metrics.RMSE(),
    random_seed=42,
    logging_level='Silent',
    objective=metrics.RMSE(),
    task_type='GPU', # training on GPU
    use_best_model=True,
    od_type='Iter',
    od_wait=20,
    **best_params,
)

# Train the model with early stopping
best_model.fit(train_pool, eval_set=val_pool)
