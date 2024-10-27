from sklearn.model_selection import train_test_split
import pandas as pd
from catboost import CatBoostRegressor, Pool, metrics, cv
import optuna

# Load the dataset after the exploratory data analysis
challenge_set_updated = pd.read_csv("../data/challenge_set_updated_v20.csv")
submission_set = pd.read_csv("../data/final_submission_set_v20.csv")
submission_set_updated = pd.read_csv("../data/submission_set_updated.csv")

dataset = pd.concat([challenge_set_updated, submission_set_updated], axis=0)

# Imputation of NaNs
columns_with_nan = dataset.isna().any()
for col in dataset.columns[columns_with_nan]:
    dataset.loc[:, col] = dataset.fillna(dataset[col].median())

df = dataset.iloc[0:challenge_set_updated.shape[0], :]

# Separating features and target variable
X = df.drop('tow', axis=1)
y = df['tow']

# Eliminating bad features (see catboost_select.ipynb)
eliminated_features = ['groundspeed_airspeed_ratio_ENR', 'temperature_9', 'wind_distance_flown_distance_ENR', 'average_humidity_DEP_40', 'vertical_rate_bins_ARR', 
        'groundspeed_flown_distance_ARR', 'arrival_quarter', 'offblock_year', 'arrival_year', 'offblock_to_arrival_day_diff', 'altitude_9', 'tas_1', 
        'is_arrival_weekend', 'adep_height_6', 'sqrd_vlof_tas', 'average_airspeed_ARR_100', 'adep_height_7', 'wind_distance_ARR_100', 'altitude_4', 
        'adep_height_1', 'groundspeed_airspeed_ratio_ARR', 'tas_8', 'specific_energy_4', 'temperature_bins_DEP', 'temperature_6', 'humidity_bins_DEP', 
        'altitude_5', 'adep_height_5', 'sqrd_tas_8', 'sqrd_tas_7', 'specific_energy_7', 'specific_energy_1', 'adep_height_4', 'sqrd_tas_6', 'tas_2', 
        'sqrd_tas_5', 'specific_energy_3', 'altitude_8', 'specific_energy_6', 'adep_height_8', 'vertical_rate_airspeed_ARR', 'altitude_2', 'sqrd_tas_1', 
        'sqrd_tas_3', 'specific_energy_8', 'sqrd_tas_9', 'temperature_8', 'groundspeed_airspeed_ratio_DEP', 'sqrd_tas_4', 'altitude_6', 
        'specific_energy_5', 'humidity_temperature_DEP', 'adep_height_2', 'altitude_7', 'adep_height_3', 'temperature_1', 'specific_energy_2', 
        'temperature_5', 'wind_distance_flown_distance_ARR', 'arrival_month', 'temperature_4', 'groundspeed_ARR_100', 'tas_4', 'arrival_minute', 
        'adep_height_9', 'altitude_groundspeed_ARR', 'altitude_3', 'temperature_7', 'airspeed_specific_energy_ENR', 'altitude_10', 'sqrd_tas_10', 
        'humidity_bins_ARR', 'specific_energy_9', 'sqrd_tas_2', 'temperature_2', 'tas_10', 'average_humidity_ENR', 'offblock_quarter', 
        'airspeed_specific_energy_DEP', 'wind_distance_flown_distance_DEP', 'tas_6', 'flown_distance_ARR_100', 'vertical_rate_airspeed_ratio_ARR', 
        'average_humidity_ARR_100', 'specific_energy_10', 'first_adep_height', 'tas_3', 'temperature_3', 'track_variation_ARR_100', 
        'is_offblock_rush_hour', 'average_temperature_ENR', 'is_arrival_rush_hour', 'average_altitude_ARR_100', 'specific_energy_ENR', 
        'groundspeed_ENR', 'is_offblock_weekend', 'Num_Engines', 'temperature_bins_ARR', 'average_temperature_ARR_100', 'kpi17_time', 
        'average_airspeed_DEP_40', 'wind_distance_ENR', 'offblock_minute', 'groundspeed_10NM', 'average_vertical_rate_ARR_100', 'vlof_tas', 
        'humidity_temperature_ARR']

cat_names = ['callsign',
            'adep', 
            'ades', 
            'aircraft_type', 
            'wtc', 
            'airline',
            'offblock_hour',
            'offblock_minute', 
            'offblock_day_of_week',
            'offblock_weekday_name',
            'offblock_month',
            'offblock_week_of_year', 
            'offblock_season', 
            'arrival_hour',
            'arrival_minute',
            'arrival_season',
            'arrival_weekday_name',
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
             

X.drop(eliminated_features, axis=1, inplace=True)

selected_cat_names = [x for x in cat_names if x in X.columns]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_pool = Pool(X_train, y_train, cat_features=selected_cat_names)
val_pool = Pool(X_val, y_val, cat_features=selected_cat_names)

def objective(trial):
    # Taken from: 
    # https://deepnote.com/app/svpino/Tuning-Hyperparameters-with-Optuna-ea1a123d-8d2f-4e20-8f22-95f07470d557
    # https://forecastegy.com/posts/catboost-hyperparameter-tuning-guide-with-optuna/
    params = {
        'learning_rate' : 0.05,
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        'random_strength': trial.suggest_float('random_strength', 10, 50),
        'depth': trial.suggest_int('depth', 4, 10),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30),
        'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 15),
    }
    
    model = CatBoostRegressor(
        iterations=20000,
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
  
    return best_rmse
    
study = optuna.create_study(study_name='catboost_tunning', storage='sqlite:///catboost.db', 
			    direction='minimize', load_if_exists=True)
study.optimize(objective, n_trials=100)

# Display the best hyperparameters found
print(f"Best trial: {study.best_trial.params}")

# Train the final model with the best parameters
best_params = study.best_trial.params
best_model = CatBoostRegressor(
    iterations=20000,
    eval_metric=metrics.RMSE(),
    random_seed=42,
    logging_level='Silent',
    objective=metrics.RMSE(),
    task_type='GPU', # training on GPU
    use_best_model=True,
    od_type='Iter',
    od_wait=50,
    **best_params,
)

# Train the model with early stopping
best_model.fit(train_pool, eval_set=val_pool)
