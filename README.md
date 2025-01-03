# Real Estate Price Prediction with Streamlit

This project focuses on analyzing real estate data and building predictive insights on house prices. The dataset contains features such as house size, number of bedrooms, and other key attributes.

## Project Overview
The goal is to explore the dataset, prepare it for analysis, visualize key trends, and build a predictive model to estimate house prices.

## Key Steps

### 1. Data Preparation
- Loaded the real estate data from a CSV file (`real_state_dataset.csv`).
  ```python
  import pandas as pd
  df = pd.read_csv('real_state_dataset.csv')
  ```
- Displayed basic dataset information including shape, column names, and data types.
  ```python
  print(df.shape)
  print(df.info())
  ```
- Inspected the first few rows using `head()`.
  ```python
  print(df.head())
  ```
- Checked for missing values using `df.isnull().sum()`.
  ```python
  print(df.isnull().sum())
  ```
- Dropped unnecessary columns: `brokered_by`, `zip_code`, and `prev_sold_date`.
  ```python
  df.drop(columns=['brokered_by', 'zip_code', 'prev_sold_date'], inplace=True)
  ```
- Removed rows with missing values using `dropna()`.
  ```python
  df.dropna(inplace=True)
  ```
- Checked for duplicate entries and removed them using `drop_duplicates()`.
  ```python
  df.drop_duplicates(inplace=True)
  ```

### 2. Exploratory Data Analysis (EDA)
- Calculated descriptive statistics (count, mean, min, max) for numerical columns using `describe()`.
  ```python
  print(df.describe())
  ```
- Analyzed the distribution of key features.
- Visualized the top 10 states with the most houses using a bar plot.
  ```python
  import matplotlib.pyplot as plt
  df['state'].value_counts().sort_values(ascending=False).head(10).plot(kind='bar')
  plt.title('Top 10 States with Most Houses')
  plt.show()
  ```

  ![image](https://github.com/user-attachments/assets/c3a63846-4c48-46a5-91cc-9f570dfc274c)

  

- Calculated average house prices by state and city.
  ```python
  avg_price_by_state = df.groupby('state')['price'].mean()
  print(avg_price_by_state)
  ```
- Displayed the correlation between numerical features and the target variable (`price`).
  ```python
  print(df.corr()['price'])
  ```

### 3. Feature Engineering and Selection
- Selected relevant features (`bed`, `bath`, `house_size`) for model building.
  ```python
  X = df[['bed', 'bath', 'house_size']]
  y = df['price']
  ```
- No additional feature engineering was performed.

### 4. Model Building and Evaluation
- Split the dataset into training and testing sets using `train_test_split`.
  ```python
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  ```
- Standardized the numerical features using `StandardScaler` to improve model performance.
  ```python
  from sklearn.preprocessing import StandardScaler
  from sklearn.linear_model import LinearRegression
  import joblib

  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)
  joblib.dump(scaler, 'scaler.pkl')
  ```
- Trained a **Linear Regression** model using the training data.
  ```python
  lr = LinearRegression()
  lr.fit(X_train, y_train)
  ```
- Made predictions on the test data and evaluated the model using **Mean Absolute Error (MAE)**.
  ```python
  from sklearn.metrics import mean_absolute_error
  lr_pred = lr.predict(X_test)
  mae = mean_absolute_error(y_test, lr_pred)
  print(f'Mean Absolute Error: {mae}')
  ```
- Saved the trained model and scaler using `joblib.dump()`.
  ```python
  joblib.dump(lr, 'model.pkl')
  ```

### 5. Streamlit Application
- A Streamlit app was developed to allow users to input house features and get a predicted price.
  ```python
  import streamlit as st
  import joblib
  import numpy as np

  scaler = joblib.load('scaler.pkl')
  model = joblib.load('model.pkl')

  st.title('House Price Prediction')

  st.divider()

  bed = st.number_input('Bedrooms', value=2 , step=1)
  bath = st.number_input('Bathrooms', value=1, step=1)
  house_size = st.number_input('House Size', value=1000, step=50)

  X = [bed, bath, house_size]

  st.divider()

  predict_btn = st.button('Predict')
  st.divider()

  if predict_btn:
      st.balloons()   
      X1 = np.array(X)
      X_array = scaler.transform([X1])
      prediction = model.predict(X_array)[0]
      st.write(f'Predicted Price: {prediction:.2f}')
  else:
      st.write('Click the button to predict the price')
  ```

## Results
- The model was evaluated using **Mean Absolute Error (MAE)**, which measures how close predictions are to actual values.
- The Streamlit app provides an interactive interface for predicting house prices.

**Result Image**


<img width="656" alt="image" src="https://github.com/user-attachments/assets/5c1e2fb8-9f8c-47c2-b7d0-97a0862605b9" />




## Future Work
- Handle outliers (if present) and perform feature scaling.
- Compare the Linear Regression model with other machine learning models (e.g., Decision Trees, Random Forests).
- Tune hyperparameters to improve model performance.
- Deploy the final model as a web application.


## Contributing
Contributions are welcome! Please fork the repository and create a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License.
