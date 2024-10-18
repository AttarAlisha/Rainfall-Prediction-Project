import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Sample dataset for demonstration purposes (replace with actual dataset)
data = {
    'Temperature': [30, 25, 40, 22, 35, 28, 31, 26],
    'Humidity': [65, 70, 85, 75, 60, 55, 68, 72],
    'Pressure': [1012, 1008, 1015, 1005, 1010, 1009, 1011, 1007],
    'Rainfall': [5, 10, 0, 15, 2, 0, 8, 12]  # Rainfall in mm
}

df = pd.DataFrame(data)

# Features and target variable
X = df[['Temperature', 'Humidity', 'Pressure']]
y = df['Rainfall']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model using pickle
with open('rainfall_model.pkl', 'wb') as file:
    pickle.dump(model, file)
