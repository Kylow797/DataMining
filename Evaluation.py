import pandas as pd
import joblib as job
from sklearn.metrics import f1_score, accuracy_score, classification_report

# Load the saved test data
test_data = pd.read_csv('/Users/kylow/Dev/Data Mining/testsplit.csv')
test_data = test_data.iloc[1:].reset_index(drop=True)  

# Separate features (X) and target (y)
X_test = test_data.drop('Target (Col 106)', axis=1)
y_test = test_data['Target (Col 106)']

# Load the saved model
best_model = job.load('best_prediction_model.joblib')

# Make predictions on the test set
y_test_pred = best_model.predict(X_test)

# Calculate performance metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred, average='binary')

# Output results
print("\nTest Set Metrics:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"F1 Score: {test_f1:.4f}")

# Print classification report for more details
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred))