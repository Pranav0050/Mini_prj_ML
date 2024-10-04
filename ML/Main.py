import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('cars.csv')  # Replace with your CSV file name

# Separate features and target
X = data.drop('price', axis=1)  # Replace 'target_column' with your target column name
y = data['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the models
linear_model = LinearRegression()
tree_model = DecisionTreeRegressor(random_state=42)

linear_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)

# Make predictions of ar prices
linear_pred = linear_model.predict(X_test)
tree_pred = tree_model.predict(X_test)

# Calculate metrics
linear_mse = mean_squared_error(y_test, linear_pred)
linear_r2 = r2_score(y_test, linear_pred)

tree_mse = mean_squared_error(y_test, tree_pred)
tree_r2 = r2_score(y_test, tree_pred)

# Print results
print("Linear Regression Results:")
print(f"Mean Squared Error: {linear_mse}")
print(f"R-squared Score: {linear_r2}")
print("\nDecision Tree Regression Results:")
print(f"Mean Squared Error: {tree_mse}")
print(f"R-squared Score: {tree_r2}")

# Determine the better model
better_model = "Linear Regression" if linear_r2 > tree_r2 else "Decision Tree Regression"
print(f"\nThe better model is: {better_model}")

# Visualize the results
plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.scatter(y_test, linear_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Linear Regression')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

plt.subplot(122)
plt.scatter(y_test, tree_pred, color='green', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Decision Tree Regression')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

plt.tight_layout()
plt.savefig('regression_comparison.png')
plt.show()

# Compare feature importances (for Decision Tree)
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': tree_model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.title('Feature Importances (Decision Tree)')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('feature_importances.png')
plt.show()