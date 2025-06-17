from flask import Flask, request, render_template, jsonify, redirect, url_for
import joblib
import pandas as pd
from PIL import Image
import io
import numpy as np
import traceback
import tempfile # Import tempfile for temporary file creation
import os       # Import os for file operations (like removing temp file)

# Assuming utils.py contains ImageFeatureExtractor
from utils import ImageFeatureExtractor

app = Flask(__name__)

# --- GLOBAL MODEL & PIPELINE LOADING ---
# These variables will hold your loaded models and pipelines
full_preprocessing_pipeline = None
Tabular_XGBoost_model = None
image_feature_extractor = None
full_multi_modal_preprocessing_pipeline = None
XGBoost_model = None # This is your multimodal model
Stacking_model = None

try:
        print("Loading tabular preprocessing pipeline...")
        full_preprocessing_pipeline = joblib.load('full_preprocessing_pipeline.joblib')
        print("Successfully loaded tabular preprocessing pipeline.")

        print("Loading tabular model...")
        Tabular_XGBoost_model = joblib.load('Tabular_XGBoost_model.joblib')
        print("Successfully loaded tabular model.")

        print("Initializing ImageFeatureExtractor...")
        image_feature_extractor = ImageFeatureExtractor()
        print("ImageFeatureExtractor: Initialized with ResNet50 and loaded for feature extraction.")

        print("Loading multimodal preprocessing pipeline...")
        full_multi_modal_preprocessing_pipeline = joblib.load('full_multi_modal_preprocessing_pipeline.joblib')
        print("Successfully loaded multimodal preprocessing pipeline.")

        print("Loading multimodal model...")
        XGBoost_model = joblib.load('XGBoost_model.joblib') # Your multimodal model
        print("Successfully loaded multimodal model.")

        print("Loading stacking model...")
        Stacking_model = joblib.load('Stacking_Regressor_model.joblib') # Your multimodal model
        print("Successfully loaded multimodal model.")

except Exception as e:
        print(f"ERROR loading models or pipelines: {e}")
        traceback.print_exc()
        # In a production app, you might want to return an error page or log heavily
        # For development, exiting is fine if models are critical
        exit()

# --- ROUTES ---

# Initial choice page
@app.route('/')
def index():
    return render_template('choice.html')

# Route for the tabular-only input form
@app.route('/tabular_form')
def show_tabular_form():
    return render_template('tabular_form.html')

# Route for the multimodal input form
@app.route('/multimodal_form')
def show_multimodal_form():
    return render_template('multimodal_form.html')


# Tabular prediction route (existing, but ensure it's correct)
@app.route('/predict_tabular', methods=['POST'])
def predict_tabular():
    try:
        # --- Debugging raw form data ---
        form_data = request.form.to_dict()
        print(f"--- Debug: Raw form_data received: {form_data}")

        # Prepare input data dictionary based on form fields
        # Ensure all columns in the dataset start with a capital letter
        input_data = {
            'Weight': [float(form_data.get('Weight', 0.0))],
            'X': [float(form_data.get('X', 0.0))],
            'Y': [float(form_data.get('Y', 0.0))],
            'Z': [float(form_data.get('Z', 0.0))],
            'Cut': [form_data.get('Cut', '')],
            'Polish': [form_data.get('Polish', '')],
            'Symmetry': [form_data.get('Symmetry', '')],
            'Clarity': [form_data.get('Clarity', '')],
            'Colour': [form_data.get('Colour', '')],
            'Fluorescence': [form_data.get('Fluorescence', '')],
            'Shape': [form_data.get('Shape', '')],
            'Colour_IsFancy': [form_data.get('Colour_IsFancy', '0')]
        }

        # Create DataFrame from input data
        input_df = pd.DataFrame(input_data)
        print(f"--- Debug: Received raw tabular input:\n{input_df}")

        # Apply preprocessing pipeline
        processed_features = full_preprocessing_pipeline.transform(input_df)
        print(f"--- Debug: Processed tabular features shape: {processed_features.shape}")

        # Make prediction
        prediction = Tabular_XGBoost_model.predict(processed_features)[0] # Get the first (and only) prediction
        print(f"--- Debug: Tabular prediction raw output: {prediction}")

        # Assuming the prediction is the final price, no inverse transform needed unless scaled
        return jsonify({'predicted_price': float(prediction)})

    except Exception as e:
        print(f"Error during tabular prediction: {e}")
        traceback.print_exc() # Print full traceback for debugging
        return jsonify({'error': 'There was a problem processing your tabular input. Please check your input.', 'details': str(e)}), 500


# New Multimodal prediction route (takes both tabular and image)
@app.route('/predict_multimodal', methods=['POST'])
def predict_multimodal():
    temp_image_path = None # Initialize to None for cleanup in finally block
    try:
        # 1. Handle Tabular Data
        form_data = request.form.to_dict()
        print(f"--- Debug: Raw form_data received for multimodal: {form_data}")

        input_data = {
            'Weight': [float(form_data.get('Weight', 0.0))],
            'X': [float(form_data.get('X', 0.0))],
            'Y': [float(form_data.get('Y', 0.0))],
            'Z': [float(form_data.get('Z', 0.0))],
            'Cut': [form_data.get('Cut', '')],
            'Polish': [form_data.get('Polish', '')],
            'Symmetry': [form_data.get('Symmetry', '')],
            'Clarity': [form_data.get('Clarity', '')],
            'Colour': [form_data.get('Colour', '')],
            'Fluorescence': [form_data.get('Fluorescence', '')],
            'Shape': [form_data.get('Shape', '')],
            'Colour_IsFancy': [form_data.get('Colour_IsFancy', '0')]
        }
        # Do NOT create DataFrame yet, we need to add the image path column

        # 2. Handle Image Data (save temporarily)
        file = None # Ensure 'file' is defined
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided for multimodal prediction'}), 400
        else:
            file = request.files['image']

        if file.filename == '':
            return jsonify({'error': 'No selected image file for multimodal prediction'}), 400

        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes))

        # --- Save image to a temporary file ---
        # Create a temporary file and get its path
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            img.save(tmp.name)
            temp_image_path = tmp.name
        print(f"--- Debug: Image saved temporarily to: {temp_image_path}")

        # 3. Create the input DataFrame for the multimodal pipeline
        # Now, add the temporary image path to your input_data
        # Make sure the column name matches what your pipeline expects for the image path!
        # Assuming your pipeline expects a column named 'Image_Path'
        input_data['image_path'] = [temp_image_path] # <--- CRITICAL: COLUMN NAME MUST MATCH
        
        # Create DataFrame with all 13 (12 tabular + 1 image path) features
        # Ensure column order matches training, if not using ColumnTransformer by name.
        # It's safer if ColumnTransformer uses column names for mapping.
        # Since 'All columns in the dataset start with a capital letter', 'Image_Path' is a good guess.
        full_input_df = pd.DataFrame(input_data)
        print(f"--- Debug: Full input DataFrame for multimodal pipeline:\n{full_input_df}")


        # 4. Apply the full_multi_modal_preprocessing_pipeline to the combined input DataFrame
        # This pipeline will now handle both tabular preprocessing and image feature extraction internally
        print("--- Debug: Applying full_multi_modal_preprocessing_pipeline...")
        processed_features_for_multimodal_model = full_multi_modal_preprocessing_pipeline.transform(full_input_df)
        print(f"--- Debug: Final processed features shape (from multimodal pipeline): {processed_features_for_multimodal_model.shape}")

        # 5. Make prediction using the multimodal model (XGBoost_model)
        print("--- Debug: Making multimodal prediction with XGBoost model...")
        multimodal_prediction_log = Stacking_model.predict(processed_features_for_multimodal_model)[0]

        # --- Apply inverse log transformation to get actual price ---
        multimodal_prediction = np.expm1(multimodal_prediction_log)

        print(f"--- Debug: Multimodal prediction raw output: {multimodal_prediction}")

        return jsonify({'predicted_price': float(multimodal_prediction)})

    except Exception as e:
        print(f"Error during multimodal prediction: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Error processing multimodal prediction', 'details': str(e)}), 500
    finally:
        # --- Cleanup: Ensure temporary file is deleted ---
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
                print(f"--- Debug: Cleaned up temporary image file: {temp_image_path}")
            except Exception as e:
                print(f"Error cleaning up temporary file {temp_image_path}: {e}")

if __name__ == '__main__':
    app.run(debug=True) # debug=True means it auto-reloads on code changes