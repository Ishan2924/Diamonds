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

```
Diamond/
├── app.py
├── utils.py
├── requirements.txt
├── .gitignore
├── full_preprocessing_pipeline.joblib
├── Tabular_XGBoost_model.joblib
├── full_multi_modal_preprocessing_pipeline.joblib
├── XGBoost_model.joblib
├── Stacked_model.joblib
├── templates/
│   ├── choice.html
│   ├── tabular_form.html
│   └── multimodal_form.html
└── static/
    └── style.css
```
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

# Output

![choice](https://github.com/user-attachments/assets/42b3270a-e51d-4870-9c77-0d9f45b118d7)

![opt2](https://github.com/user-attachments/assets/3f090499-da13-4571-93bc-b9ee0a6a5af8)

![ans2](https://github.com/user-attachments/assets/25cc2d56-ff01-47fd-be40-c4fa7b7ac0da)

![opt1](https://github.com/user-attachments/assets/3f81038e-963d-4459-b72c-fd221004d72f)

![ans1](https://github.com/user-attachments/assets/3b71e46e-4680-4c72-8b9b-47c598027543)

