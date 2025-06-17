# Diamond Price Predictor (Multimodal)

This project implements a sophisticated Diamond Price Predictor web application using Flask, capable of estimating diamond prices based on either **tabular features alone** or a **combination of tabular features and a diamond image**. It leverages a multimodal machine learning approach with a custom image feature extractor and a robust stacked regressor model.

## Features

* **Multimodal Prediction:** Predicts diamond prices using both traditional tabular characteristics (carat, cut, clarity, etc.) and visual features extracted from an uploaded image.
* **Tabular-Only Prediction:** Offers a separate prediction path for users who only have tabular data.
* **Intuitive Web Interface:** A clean, modern, and aesthetically pleasing user interface built with HTML, CSS, and Tailwind CSS, providing a seamless user experience.
* **Robust Preprocessing:** Utilizes `scikit-learn` pipelines for comprehensive data preprocessing, including imputation, scaling, and categorical encoding.
* **Image Feature Extraction:** Employs a pre-trained `ResNet50` convolutional neural network (`tensorflow.keras.applications`) to extract rich features from diamond images.
* **Stacked Regression Model:** Leverages a `Stacked Regressor` for superior prediction accuracy, trained on log-transformed target variables to handle data skewness and ensure positive price predictions.
* **Dependency Management:** Includes a `requirements.txt` for easy environment setup and reproducibility.

## Technologies Used

* **Backend:** Python 3.x, Flask
* **Machine Learning:**
    * `scikit-learn==1.6.1`
    * `tensorflow==2.19.0`
    * `keras==3.10.0`
    * `xgboost==3.0.2`
    * `numpy==1.26.4`
    * `pandas==2.2.3`
    * `joblib==1.5.1` (for model persistence)
    * `Pillow==11.2.1` (for image processing)
* **Frontend:** HTML5, CSS3, Tailwind CSS (CDN)
* **Package Management:** `pip`
* **Version Control:** `Git` / `GitHub`

*(Full list of dependencies available in `requirements.txt`)*

## Project Structure

Diamond/
├── app.py                     # Flask application main file
├── utils.py                   # Contains ImageFeatureExtractor and preprocessing transformers
├── requirements.txt           # Python dependencies
├── .gitignore                 # Specifies files/folders to ignore (e.g., venv/)
├── full_preprocessing_pipeline.joblib       # Tabular-only preprocessing pipeline
├── Tabular_XGBoost_model.joblib             # Tabular-only trained model
├── full_multi_modal_preprocessing_pipeline.joblib # Multimodal preprocessing pipeline (includes image path handling)
├── XGBoost_model.joblib                     # Multimodal trained model (XGB Regressor)
├── Stacked_model.joblib                     # Multimodal trained model (XGB Regressor)
├── templates/                 # HTML templates for Flask
│   ├── choice.html            # Landing page for input method selection
│   ├── tabular_form.html      # Form for tabular-only input
│   └── multimodal_form.html   # Form for tabular + image input
└── static/                    # Static assets like CSS
└── style.css              # Custom CSS styles

## Setup and Run Locally

Follow these steps to get the project up and running on your local machine:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/Diamond.git](https://github.com/your-username/Diamond.git) # Replace with your repo URL
    cd Diamond
    ```

2.  **Create and Activate a Virtual Environment:**
    It's highly recommended to use a virtual environment to manage dependencies.
    * **Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    * **macOS / Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install Dependencies:**
    Once your virtual environment is activated, install all required packages using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Acquire Trained Models and Pipelines:**
    Ensure you have the following `.joblib` files in your project's root directory. These are the trained models and preprocessing pipelines:
    * `full_preprocessing_pipeline.joblib`
    * `Tabular_XGBoost_model.joblib`
    * `full_multi_modal_preprocessing_pipeline.joblib`
    * `XGBoost_model.joblib`

    *(Due to size, the models and pipelines are not uploaded here, please train your own models with the training data set)*

5.  **Run the Flask Application:**
    With your virtual environment still active, run the Flask app:
    ```bash
    python app.py
    ```
    The application will start, and you should see output indicating that it's running on a specific address (e.g., `http://127.0.0.1:5000`).

## Usage

1.  **Access the Application:** Open your web browser and navigate to the address shown in your terminal (e.g., `http://127.0.0.1:5000`).

2.  **Choose Input Method:** You will be presented with an option to choose between "With Image (Multimodal)" or "Data Only".

3.  **Enter Data & Predict:**
    * **Tabular:** Fill in all the diamond's tabular characteristics and click "Predict Tabular Price".
    * **Multimodal:** Fill in the tabular characteristics AND upload an image of the diamond. Click "Predict Multimodal Price".

4.  **View Prediction:** The predicted price will be displayed on the page.

## Challenges & Learning

This project presented several common machine learning and deployment challenges:

* **Dependency Management:** Navigating conflicting `scikit-learn` versions across different notebooks was a significant hurdle, emphasizing the importance of precise `requirements.txt` and isolated virtual environments.

* **Multimodal Input Handling:** Integrating disparate data types (tabular and image) required careful pipeline design. The `ColumnTransformer`'s expectation of an `image_path` column necessitated temporarily saving uploaded images to disk during inference to match the training pipeline's input structure.

* **Target Variable Transformation:** Initial negative price predictions highlighted the necessity of `np.log1p()` transformation on the skewed 'Price' target variable during training, followed by `np.expm1()` inverse transformation during inference, to ensure sensible, positive price outputs.

* **Data Sparsity in High-Value Range:** Despite a high overall R2 score (0.97 for the Stacked Regressor), the model demonstrated a tendency to underpredict extremely high diamond prices. This is attributed to the sparse distribution of high-value diamonds in the training dataset, a common challenge in real-world skewed data, where models learn best from the dense regions. This limitation was acknowledged as an area for future work rather than a flaw in the current implementation.

## Future Enhancements

* **Expand Dataset for High Values:** Acquire more high-priced diamond data to improve prediction accuracy at the extreme ends of the price spectrum.

* **Fine-tune CNN for Diamond Features:** Experiment with fine-tuning the `ResNet50` model on a dataset of diamond images specifically for more relevant visual feature extraction.

* **Real-time Image Preprocessing:** Explore in-memory image processing solutions to avoid temporary file storage if scalability becomes a concern.

* **User Authentication:** Implement user login/registration.

* **Prediction History:** Allow users to save and review past predictions.

* **Deployment to Cloud:** Deploy the Flask application to a cloud platform (e.g., Google Cloud Run, AWS Elastic Beanstalk) for public access.


