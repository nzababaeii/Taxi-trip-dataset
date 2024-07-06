# Taxi-trip-dataset
# Taxi Trip Fare Prediction

This project aims to predict taxi trip fares based on various features such as trip distance, time of day, and pickup and dropoff locations. The dataset used for this project is from Kaggle.

## Dataset

The dataset used for this project can be downloaded from Kaggle. The dataset contains information about taxi trips including fare amount, trip distance, pickup and dropoff locations, and more.

[Kaggle Dataset: Taxi Trip Dataset](https://www.kaggle.com/datasets/anandshaw2001/taxi-dataset)

## Project Steps

1. **Download the Dataset:**
   - The first step is to download the dataset from Kaggle. Make sure you have a Kaggle account and have set up your Kaggle API token. You can download the dataset using the following command:

     ```bash
     kaggle datasets download -d anandshaw2001/taxi-dataset
     ```

2. **Install Dependencies:**
   - Install the necessary Python packages. You can do this using pip:

     ```bash
     pip install -r requirements.txt
     ```

3. **Load and Preprocess Data:**
   - Load the dataset and preprocess it. This includes handling missing values, encoding categorical variables, and scaling numerical features.

4. **Feature Engineering:**
   - Perform feature engineering to create new features from the existing data. This may include creating features like trip duration, average speed, and more.

5. **Train Machine Learning Models:**
   - Train various machine learning models to predict the fare amount. This includes models like Linear Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, and XGBoost.

6. **Evaluate Models:**
   - Evaluate the performance of the trained models using metrics like Mean Squared Error (MSE) and R² score. Compare the performance of different models.

7. **Model Comparison:**
   - Compare the performance of different models and visualize the results using plots.

## Directory Structure

```plaintext
├── my_project
│   ├── data
│   │   └── taxi-dataset.csv
│   ├── notebooks
│   │   └── Taxi.ipynb
│   ├── src
│   │   ├── data_preprocessing.py
│   │   ├── feature_engineering.py
│   │   ├── model_training.py
│   │  └── model_evaluation.py
│   └── README.md
