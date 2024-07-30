import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# Loading data
gold_data = pd.read_csv('gld_price_data.csv', index_col=0)

# Data preprocessing
gold_data.index = pd.to_datetime(gold_data.index)
gold_data.fillna(method='ffill', inplace=True)

# Adding moving averages
gold_data['SPX_MA20'] = gold_data['SPX'].rolling(window=20).mean()
gold_data['USO_MA20'] = gold_data['USO'].rolling(window=20).mean()
gold_data['SLV_MA20'] = gold_data['SLV'].rolling(window=20).mean()
gold_data['EUR/USD_MA20'] = gold_data['EUR/USD'].rolling(window=20).mean()
gold_data.dropna(inplace=True)

# Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(gold_data.drop(['GLD'], axis=1))
scaled_gold_data = pd.DataFrame(scaled_features, index=gold_data.index, columns=gold_data.columns[:-1])
scaled_gold_data['GLD'] = gold_data['GLD']

# Visualization of data
corr_mat = gold_data.corr()
plt.figure(figsize=(8,8))
sns.heatmap(corr_mat, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')
plt.show()

sns.distplot(gold_data['GLD'], color='green')
plt.show()

# Model preparation
X = gold_data.drop(['GLD'], axis=1)
Y = gold_data['GLD']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Hyperparameter tuning using Grid Search for RandomForest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_rf = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid_rf, cv=3, n_jobs=-1, verbose=2)
grid_search_rf.fit(X_train, Y_train)

# Hyperparameter tuning using Grid Search for GradientBoosting
param_grid_gb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.05],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}
grid_search_gb = GridSearchCV(estimator=GradientBoostingRegressor(), param_grid=param_grid_gb, cv=3, n_jobs=-1, verbose=2)
grid_search_gb.fit(X_train, Y_train)

# Evaluate and compare models
best_model_rf = grid_search_rf.best_estimator_
test_data_prediction_rf = best_model_rf.predict(X_test)
best_model_gb = grid_search_gb.best_estimator_
test_data_prediction_gb = best_model_gb.predict(X_test)

models = {
    'Random Forest': best_model_rf,
    'Gradient Boosting': best_model_gb
}

# R squared error
error_score_rf = metrics.r2_score(Y_test, test_data_prediction_rf)
error_score_gb = metrics.r2_score(Y_test, test_data_prediction_gb)
print(f"Random Forest R squared error : {error_score_rf}")
print(f"Gradient Boosting R squared error : {error_score_gb}")

# Additional metrics
mae_rf = metrics.mean_absolute_error(Y_test, test_data_prediction_rf)
mse_rf = metrics.mean_squared_error(Y_test, test_data_prediction_rf)
rmse_rf = np.sqrt(mse_rf)
print(f"Random Forest MAE: {mae_rf}, MSE: {mse_rf}, RMSE: {rmse_rf}")

mae_gb = metrics.mean_absolute_error(Y_test, test_data_prediction_gb)
mse_gb = metrics.mean_squared_error(Y_test, test_data_prediction_gb)
rmse_gb = np.sqrt(mse_gb)
print(f"Gradient Boosting MAE: {mae_gb}, MSE: {mse_gb}, RMSE: {rmse_gb}")

# Selecting best model
if error_score_rf > error_score_gb:
    best, best_model = "RF", best_model_rf 
else:
    "GB", best_model_gb

# Feature importance
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
plt.tight_layout()
plt.show()

# Actual vs Predicted
plt.figure(figsize=(14, 6))
plt.plot(Y_test.values, color='blue', label='Actual Value')
if best == "RF": 
    plt.plot(test_data_prediction_rf, color='green', label='Random Forest Predicted Value')
elif best == "GB":
    plt.plot(test_data_prediction_gb, color='green', label='Gradient Boosting Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()

# Save model
import joblib
joblib.dump(best_model, 'gold_price_model.pkl')