import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder # <--- ADD/ENSURE THESE LINES ARE HERE
import tensorflow as tf
import traceback
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input



class ImageFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, target_size=(224, 224), model_name='ResNet50'):
        self.target_size = target_size
        self.model_name = model_name
        self.model = None
        self.preprocess_input_fn = None

        # --- MODIFIED PART: Check if tf is not None instead of _tf_available ---
        if tf is not None:
            print(f"ImageFeatureExtractor: Initializing with {model_name}.")
            if model_name == 'ResNet50':
                base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
                self.model = Model(inputs=base_model.input, outputs=base_model.output)
                self.preprocess_input_fn = resnet_preprocess_input
                print(f"ImageFeatureExtractor: Loaded {model_name} for feature extraction.")
            else:
                raise ValueError(f"Model '{model_name}' not supported or not implemented yet.")
        else:
            print("ImageFeatureExtractor: TensorFlow not available. This extractor will return dummy features.")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        image_paths = X.iloc[:, 0].tolist()

        features = []
        # --- MODIFIED PART: Check self.model and self.preprocess_input_fn directly ---
        if self.model and self.preprocess_input_fn: # Only run if TensorFlow model was loaded successfully in __init__
            for path in image_paths:
                try:
                    # --- YOU NEED TO ENSURE THESE LINES ARE CORRECT FOR YOUR IMAGE PATHS ---
                    img = image.load_img(path, target_size=self.target_size)
                    img_array = image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = self.preprocess_input_fn(img_array) # Use the stored preprocessing function

                    cnn_features = self.model.predict(img_array, verbose=0)[0]
                    features.append(cnn_features)

                except Exception as e:
                    print(f"Error processing image {path}: {e}. Returning zeros for this image.")
                    features.append(np.zeros(2048))
        else:
            print("ImageFeatureExtractor: Returning dummy features (model not loaded).")
            for _ in image_paths:
                features.append(np.zeros(2048))

        return np.array(features)
    
    def extract_features_from_pil_image(self, pil_image):
        """
        Extracts features from a single PIL Image object directly (in-memory).
        This is suitable for web uploads.
        """
        if self.model and self.preprocess_input_fn:
            try:
                # Resize the PIL image, convert to array, expand dims, and preprocess
                img_array = image.img_to_array(pil_image.resize(self.target_size))
                img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
                img_array = self.preprocess_input_fn(img_array) # Apply model-specific preprocessing

                # Predict features using the loaded model
                cnn_features = self.model.predict(img_array, verbose=0)[0]
                return cnn_features
            except Exception as e:
                print(f"Error extracting features from PIL image: {e}. Returning zeros.")
                traceback.print_exc()
                return np.zeros(2048) # Ensure consistent output shape in case of error
        else:
            print("ImageFeatureExtractor: Returning dummy features (model not loaded for PIL image).")
            return np.zeros(2048) # Ensure consistent output shape if TensorFlow not available

    def get_feature_names_out(self, input_features=None):
        output_feature_dim = 2048
        if self.model and hasattr(self.model, 'output_shape') and len(self.model.output_shape) > 1:
             output_feature_dim = self.model.output_shape[-1]
        return [f"img_feature_{i}" for i in range(output_feature_dim)]
# --- Define Preprocessing Transformers ---

# 3.1 Numerical Transformer: Impute with median, then Scale
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 3.2 Ordinal Categorical Transformer: Impute with most frequent, then Ordinal Encode
# These orders must match the EXACT string categories in your DataFrame
quality_order_common = ['F', 'GD', 'VG', 'EX']
clarity_order = ['I2', 'I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
colour_order = [
    'Y-Z', 'W-X', 'U-V', 'S-T', 'Q-R', 'O-P',
    'N', 'M', 'L', 'K', 'J', 'I', 'H', 'G', 'F', 'E', 'D'
]

ordinal_encoder_categories = [
    quality_order_common, # For 'Cut'
    quality_order_common, # For 'Polish'
    quality_order_common, # For 'Symmetry'
    clarity_order,        # For 'Clarity'
    colour_order          # For 'Colour'
]

ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal_encoder', OrdinalEncoder(categories=ordinal_encoder_categories,
                                       handle_unknown='use_encoded_value',
                                       unknown_value=-1))
])

# 3.3 Nominal Categorical Transformer: Impute with most frequent, then One-Hot Encode
nominal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 3.4 Image Transformer
image_transformer = Pipeline(steps=[
    # Note: ImageFeatureExtractor doesn't need an imputer because it handles missing/errors internally
    ('image_extractor', ImageFeatureExtractor())
])