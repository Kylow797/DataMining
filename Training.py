import pandas as pd
import joblib as job
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report

data = pd.read_csv('/Users/kylow/Dev/Data Mining/DM_Project_24.csv')
data = data.iloc[1:].reset_index(drop=True)  
X = data.drop('Target (Col 106)',axis=1)
Y = data['Target (Col 106)']




#Pipeline
pipeline = Pipeline(steps=[
    #Median Imputer
    ('imputer', SimpleImputer(strategy='mean')),  
    #Standard Scaler
    ('scaler', StandardScaler()),
    #KNN                   
    ('classifier', KNeighborsClassifier())          
])

param_grid = {
    #Number of Neighbours
    'classifier__n_neighbors': [3,5,7,9,11,13],          
    #Weight
    'classifier__weights': ['uniform','distance'],    
    #Distance 
    'classifier__metric': ['euclidean','minkowski','manhattan','chebyshev'],
    #algo
    'classifier__algorithm':['auto','ball_tree','kd_tree','brute'],
    #leaf
    'classifier__leaf_size': [10,20,30],
    #p
    'classifier__p': [1,2,3]
}


grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='f1', n_jobs=-1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=79)

test_data = pd.concat([X_test, y_test], axis=1)
test_data.to_csv('/Users/kylow/Dev/Data Mining/testsplit.csv', index=False)

grid_search.fit(X_train, y_train)


print("Best Hyperparameters: ", grid_search.best_params_)
best_model = grid_search.best_estimator_


##########################----UNCOMMENT TO UPDATE TESTING MODEL----##################
#job.dump(best_model, 'best_prediction_model.joblib')
##########################----UNCOMMENT TO UPDATE TESTING MODEL----##################


# Output results
print("Training Completed: best model saved")
