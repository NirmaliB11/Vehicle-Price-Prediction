import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('synthetic_vehicle_data.csv')
    return data

data = load_data()

# Sidebar for user input
st.sidebar.header("Enter Vehicle Details")

def get_user_input():
    make = st.sidebar.selectbox('Make', data['make'].unique())
    model = st.sidebar.selectbox('Model', data['model'].unique())
    year = st.sidebar.slider('Year', 2000, 2023, 2015)
    mileage = st.sidebar.number_input('Mileage (in km)', 0, 500000, 40000)
    condition = st.sidebar.slider('Condition (1 - Poor, 5 - Excellent)', 1, 5, 3)
    
    # Dictionary of user input
    user_data = {
        'make': make,
        'model': model,
        'year': year,
        'mileage': mileage,
        'condition': condition
    }
    features = pd.DataFrame(user_data, index=[0])
    return features

user_input = get_user_input()

# Display user input
st.subheader('Vehicle Details Entered:')
st.write(user_input)

# Preprocessing
X = data[['year', 'mileage', 'condition']]
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Prediction
prediction = model.predict(user_input[['year', 'mileage', 'condition']])

# Display Prediction
st.subheader('Predicted Price:')
st.write(f'${prediction[0]:,.2f}')

# Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.subheader('Model Performance:')
st.write(f'Mean Squared Error: {mse:.2f}')
