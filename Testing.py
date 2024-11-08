import pandas as pd
import joblib as job
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report

testdata = pd.read_csv('/Users/kylow/Dev/Data Mining/test_data.csv')
best_model = job.load('best_prediction_model.joblib')
testdatapred = best_model.predict(testdata)
tdpdf = pd.DataFrame(testdatapred)

tdpdf = tdpdf.iloc[1:].reset_index(drop=True)


#Change to preferred destination
tdpdf.to_csv('/Users/kylow/Dev/Data Mining/testdatapredictions.csv', index=False)