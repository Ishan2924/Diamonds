import pandas as pd
import numpy as np
import joblib

XGBOOST_MODEL_PATH = 'Tabular_XGBoost_model.joblib'
PREPROCESSING_PIPELINE_PATH = 'full_preprocessing_pipeline.joblib'

if __name__ == '__main__':
    print("Start")

    loaded_preprocessing_pipeline = None
    loaded_model = None

    try:
        loaded_preprocessing_pipeline = joblib.load(PREPROCESSING_PIPELINE_PATH)
        print(f"Succesfully loaded pipeline from: '{PREPROCESSING_PIPELINE_PATH}'")

        loaded_model = joblib.load(XGBOOST_MODEL_PATH)
        print(f"Succesfullt loaded model from : '{XGBOOST_MODEL_PATH}'" )

    except FileNotFoundError as e:
        print(f"file not found {e}")
        exit()

    except Exception as e:
        print(f"Unexpected error {e}")


    new_raw_diamond_data_points = {
        'Weight' : [0.90, 1.15, 0.55, 1.00, 0.75],
        'X': [6.1, 6.7, 5.2, 6.4, 5.6],
        'Y': [6.0, 6.6, 5.1, 6.3, 5.5],
        'Z': [3.7, 4.1, 3.2, 3.9, 3.4],
        'Cut' : ['EX', 'VG', 'EX','GD','VG'],
        'Polish': ['EX', 'GD', 'EX', 'VG', 'VG'],
        'Symmetry': ['EX', 'VG', 'EX', 'GD', 'VG'],
        'Clarity': ['VS1', 'SI2', 'IF', 'VS2', 'SI2'],
        'Fluorescence': ['M', 'M', 'N', 'F', 'F'],
        'Colour' : ['H','F','D','G','J'],
        'Depth': [61.9, 60.8, 62.0, 61.5, 63.5],
        'Table': [56.0, 58.0, 57.0, 59.0, 61.0],

        'Shape' : ['Round', 'Cushion', 'Heart', 'Emerald', 'Round'],
        'Colour_IsFancy' : [ 0, 0,1, 0, 0]

    }

    new_raw_df = pd.DataFrame(new_raw_diamond_data_points)
    print("\n--- New Raw Diamond Data Input ---")
    print(new_raw_df)

    print("\n--- Running Preprocessing Pipeline on New Data ---")
    try:
        # The pipeline handles all transformation steps automatically
        processed_new_data = loaded_preprocessing_pipeline.transform(new_raw_df)
        print("Successfully preprocessed new data.")
        # Depending on your pipeline's last step, this might be a NumPy array or DataFrame.
        # If it's a NumPy array, converting to DataFrame is useful for inspection:
        # print(pd.DataFrame(processed_new_data, columns=loaded_preprocessing_pipeline.get_feature_names_out())) # Requires sklearn 1.0+ for get_feature_names_out
        print(f"Shape of processed data: {processed_new_data.shape}")

    except Exception as e:
        print(f"\nERROR during data preprocessing: {e}")
        print("Please ensure your input data matches the format expected by your 'full_preprocessing_pipeline'.")
        print("Check if column names and data types are consistent with your training data.")
        exit()

    print("\n--- Making Predictions ---")
    try:
        predicted_prices = loaded_model.predict(processed_new_data)
    except Exception as e:
        print(f"\nERROR during prediction: {e}")
        print("This could be due to mismatched features between processed data and model expectation.")
        print(
            "Ensure your preprocessing pipeline outputs features in the exact format (order and count) the model expects.")
        exit()

    print("\n--- Predicted Prices ---")
    for i, pred_price in enumerate(predicted_prices):
        print(f"Diamond {i + 1} (Weight: {new_raw_df.loc[i, 'Weight']}, Cut: {new_raw_df.loc[i, 'Cut']}, Colour: {new_raw_df.loc[i, 'Colour']}): Predicted Price = ${pred_price:.2f}")