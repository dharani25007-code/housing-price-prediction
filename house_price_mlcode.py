import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option("display.max_rows",10000)
pd.set_option("display.max_columns",6)
df = pd.read_csv(r'C:\Users\DHARANIDHARAN\Downloads\housing.csv')
print(df)

# Step 1: Load dataset
df = pd.read_csv(r'C:\Users\DHARANIDHARAN\Downloads\housing.csv')

# Step 2: Display first few rows
print("Dataset Preview:")
print(df.head())

# Step 3: Clean and prepare data
# Fix column names (remove spaces and special chars)
df.columns = df.columns.str.strip().str.replace(' ', '_')

# Check column names
print("\nColumns:", df.columns.tolist())

# Step 4: Define independent (X) and dependent (y) variables
# Predicting 'MEDV' — adjust if target column is different
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# Convert categorical columns to numeric
X = pd.get_dummies(X, drop_first=True)

# Fill missing numeric values (if any)
X = X.fillna(X.mean(numeric_only=True))
y = y.fillna(y.mean())

# Step 5: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate model
print("\nModel Performance:")
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Step 9: Compare actual vs predicted
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nActual vs Predicted:")
print(results.head())

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")
plt.show()
residuals = y_test - y_pred

plt.figure(figsize=(6,4))
plt.scatter(y_pred, residuals)
plt.axhline(y=0)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals (Errors)")
plt.title("Residual Plot")
plt.show()
plt.figure(figsize=(6,4))
sns.histplot(residuals, kde=True)
plt.title("Distribution of Prediction Errors")
plt.xlabel("Error")
plt.show()
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()
plt.figure(figsize=(10,4))
plt.plot(y_test.values[:50], label="Actual")
plt.plot(y_pred[:50], label="Predicted")
plt.legend()
plt.title("Actual vs Predicted (First 50 Samples)")
plt.show()

dt_model = DecisionTreeRegressor(max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("Decision Tree R²:", r2_score(y_test, y_pred_dt))
print("Decision Tree MSE:", mean_squared_error(y_test, y_pred_dt))
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_dt)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Decision Tree: Actual vs Predicted")
plt.show()
plt.figure(figsize=(20,10))
plot_tree(
    dt_model,
    feature_names=X.columns,
    filled=True,
    rounded=True
)
plt.title("Decision Tree Structure")
plt.show()