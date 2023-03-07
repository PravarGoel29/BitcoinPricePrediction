import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('BTC-USD.csv')

# Feature engineering
data['Date'] = pd.to_datetime(data['Date'])
data['Week'] = data['Date'].apply(lambda x: x.isocalendar().week)
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['Year'] = data['Date'].dt.year

# Split into train and test sets
X = data.drop(['Date', 'Close'], axis=1)
y = data['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training and hyperparameter tuning
model = RandomForestRegressor(random_state=42)
n_estimators = [100, 200, 300, 400, 500]
max_depth = [5, 10, 15, 20, 25, None]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
hyperparameters = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}
grid_search = GridSearchCV(model, hyperparameters, cv=5)
grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_

# Model evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean Squared Error:', mse)
print('R-squared:', r2)

# Feature importance plot
feature_importances = pd.DataFrame(model.feature_importances_,
                                   index=X_train.columns,
                                   columns=['Importance'])
feature_importances = feature_importances.sort_values('Importance', ascending=False)
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y=feature_importances.index, data=feature_importances, color='blue')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 8))
sns.scatterplot(x=y_pred, y=residuals, color='blue')
plt.axhline(y=0, color='r', linestyle='-')
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

# Time series plot
plt.figure(figsize=(20, 10))
sns.lineplot(x='Date', y='Close', data=data, color='blue')
plt.title('Bitcoin Close Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
