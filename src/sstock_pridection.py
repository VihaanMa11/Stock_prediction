# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the stock data
file_path = "faang_stocks.csv"  # Replace with your dataset path
data = pd.read_csv(file_path)

# Step 2: Process the data
# Convert 'Date' column to datetime and set it as the index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Calculate daily returns for one stock (e.g., AAPL)
data['GOOGL_Returns'] = data['GOOGL'].pct_change() * 100

# Create moving averages as features
data['GOOGL_MA_5'] = data['GOOGL_Returns'].rolling(window=5).mean()
data['GOOGL_MA_10'] = data['GOOGL_Returns'].rolling(window=10).mean()

# Drop rows with missing values (NaN)
data.dropna(inplace=True)

# Step 3: Prepare features (X) and target (y)
X = data[['GOOGL_MA_5', 'GOOGL_MA_10']]  # Moving averages
y = data['GOOGL_Returns']  # Next day's returns

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Step 5: Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# Step 8: Visualize actual vs predicted returns
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label="Actual Returns", color="blue")
plt.plot(y_pred, label="Predicted Returns", color="red", linestyle="dashed")
plt.title("Actual vs Predicted Stock Returns (GOOGL)")
plt.xlabel("Time")
plt.ylabel("Returns (%)")
plt.legend()
plt.show()
