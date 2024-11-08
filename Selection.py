from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.datasets import load_iris
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

# Load dataset
data = pd.read_csv('/Users/kylow/Dev/Data Mining/DM_Project_24.csv')
print(data.columns)
data = data.drop(index=0)
X = data.drop('Target (Col 106)',axis=1)
Y = data['Target (Col 106)']



# Define preprocessing and models
preprocessing_options = {
    #Impute first then Normalize
    'Method A': Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),           # Mean imputation
        ('scaler', StandardScaler())                           # Standard scaling
    ]),
    'Method B': Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),         # Median imputation
        ('scaler', MinMaxScaler())                             # Min-max scaling
    ]),
    'Method C': Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),           # Mean imputation
        ('scaler', StandardScaler())                              # Robust scaling
    ]),
    'Method D': Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Most frequent imputation
        ('scaler', MinMaxScaler())                            # Standard scaling
    ]),
    #Normalize first
    'Method E': Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('imputer', SimpleImputer(strategy='mean')),                                            # Min-max scaling
    ]),
    'Method F': Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('imputer', SimpleImputer(strategy='median')),                                            # Min-max scaling
    ]),
    'Method G': Pipeline(steps=[
        ('scaler', MinMaxScaler()),
        ('imputer', SimpleImputer(strategy='mean')),                                            # Min-max scaling
    ]),
    'Method H': Pipeline(steps=[
        ('scaler', MinMaxScaler()),
        ('imputer', SimpleImputer(strategy='median')),                                            # Min-max scaling
    ]),

}

models = {
    'KNN': KNeighborsClassifier(algorithm='auto', leaf_size=10, metric='euclidean', n_neighbors=11, p=1, weights='uniform'),
}

# Outer CV for model selection
outer_cv = KFold(n_splits=5, shuffle=True, random_state=79)

results = {}
f1results = {}

# Evaluate each combination of model and preprocessing method
for preprocessing_name, preprocessing in preprocessing_options.items():
    for model_name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessing', preprocessing),
                                    ('model', model)])

        # Inner CV for hyperparameter tuning (you can define param_grid as needed)
        param_grid = {}  # Define hyperparameter grid if necessary
        grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='f1', error_score='raise')
        
        # Perform outer CV to evaluate the model
        grid_search.fit(X, Y) 
        scores = cross_val_score(grid_search, X, Y, cv=outer_cv, scoring='f1')
        
        
        # Store results
        results[(preprocessing_name, model_name)] = scores.mean()

        print(preprocessing_name, model_name,scores.mean())

# Display results
best_combination = max(results, key=results.get)
print("Best combination:", best_combination, "with score:", results[best_combination])