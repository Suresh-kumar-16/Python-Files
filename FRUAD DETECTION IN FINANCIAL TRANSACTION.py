import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Load current transaction data
current_data = pd.read_csv('current_financial_transactions.csv')


# Load historical transaction data
historical_data = pd.read_csv('historical_financial_transactions.csv')


# Merge current and historical transaction data data = pd.concat([current_data, historical_data])

# Data preprocessing
 
# Handling missing values data.fillna(0, inplace=True)

# Feature engineering
# Create new features, drop irrelevant columns, etc. # For example:
# data['transaction_hour'] = pd.to_datetime(data['transaction_time']).dt.hour # data.drop(columns=['transaction_time'], inplace=True)

# Encoding categorical variables
data = pd.get_dummies(data, columns=['merchant_type'], drop_first=True)


# Split data into features (X) and target variable (y) X = data.drop(columns=['fraudulent'])
y = data['fraudulent']


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize pipeline for model selection and training pipeline = Pipeline([
('scaler', StandardScaler()), # Standardize features
('classifier', RandomForestClassifier(random_state=42)) # Random Forest Classifier
])


# Define hyperparameters for grid search param_grid = {
'classifier n_estimators': [100, 200, 300],
'classifier max_depth': [None, 10, 20],
 
'classifier min_samples_split': [2, 5, 10],
'classifier min_samples_leaf': [1, 2, 4]
}


# Perform grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1) grid_search.fit(X_train, y_train)

# Get best model from grid search best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred) print("Accuracy:", accuracy)

# Generate classification report print("Classification Report:") print(classification_report(y_test, y_pred))

# Generate confusion matrix print("Confusion Matrix:") print(confusion_matrix(y_test, y_pred))

# Further analysis
# Feature importance
feature_importance = best_model.named_steps['classifier'].feature_importances_
 
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance}) importance_df = importance_df.sort_values(by='Importance', ascending=False) print("\nFeature Importance:")
print(importance_df)


# Visualize feature importance import matplotlib.pyplot as plt import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10)) plt.title('Top 10 Features Importance')
plt.xlabel('Importance') plt.ylabel('Feature') plt.show()
