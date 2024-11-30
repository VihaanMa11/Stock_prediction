# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset (replace 'faang_stocks.csv' with your dataset path)
file_path = "faang_stocks.csv"
data = pd.read_csv(file_path)

# Convert 'Date' column to datetime and set it as the index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Calculate daily returns for a specific stock (e.g., AAPL)
data['AAPL_Returns'] = data['AAPL'].pct_change() * 100

# Add moving averages as features
data['AAPL_MA_5'] = data['AAPL_Returns'].rolling(window=5).mean()
data['AAPL_MA_10'] = data['AAPL_Returns'].rolling(window=10).mean()

# Drop NaN values created by rolling windows
data.dropna(inplace=True)

# Define features and target variable
X = data[['AAPL_MA_5', 'AAPL_MA_10']]  # Predictors
y = data['AAPL_Returns']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# Plot actual vs. predicted returns
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label="Actual Returns", color="blue")
plt.plot(y_pred, label="Predicted Returns", color="red", linestyle="dashed")
plt.title("Actual vs. Predicted Stock Returns (AAPL)")
plt.xlabel("Time")
plt.ylabel("Returns (%)")
plt.legend()
plt.show()
