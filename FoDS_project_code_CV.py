import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold  

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc)
import os
from sklearn.ensemble import RandomForestClassifier  

from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LassoCV
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

from sklearn import tree
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv(
    filepath_or_buffer="oral_cancer_prediction_dataset.csv",
    dtype={
        "Country": "category", "Gender": "category", "Tobacco Use": "category", "Alcohol Consumption": "category", "HPV Infection": "category",
        "Betel Quid Use": "category", "Chronic Sun Exposure": "category", "Poor Oral Hygiene": "category", "Diet (Fruits & Vegetables Intake)": "category",
        "Family History of Cancer": "category", "Compromised Immune System": "category", "Oral Lesions": "category", "Unexplained Bleeding": "category",
        "Difficulty Swallowing": "category", "White or Red Patches in Mouth": "category", "Cancer Stage": "category",  "Treatment Type": "category",
        "Early Diagnosis": "category"
    }
)


######### Methods ###########
##missing values
print("Number of missing values:", data.isna().sum(axis=1).sum()) #output is 0, so we have no missing date --> no special handling needed

##changing label oral cancer from true/false to 1/0
data["Oral Cancer (Diagnosis)"] = data["Oral Cancer (Diagnosis)"].map({"No": 0, "Yes": 1})

##first dropping all columns giving a hint to the outcome
data_ = data.drop(["ID", "Early Diagnosis", "Treatment Type", "Cancer Stage", "Survival Rate (5-Year, %)", "Cost of Treatment (USD)", "Economic Burden (Lost Workdays per Year)" , "Tumor Size (cm)"], axis=1)
#print(data_.columns)

##one-hot-encoding
cate_cols = data_.columns[data_.dtypes == "category"]
num_cols = data_.columns[data_.dtypes != "category"]
num_cols_new = num_cols.drop(["Oral Cancer (Diagnosis)"])
data_encoded = pd.get_dummies(data_, prefix=cate_cols, columns=cate_cols, dtype=int)
##construct features and labels
y = data_["Oral Cancer (Diagnosis)"]
X = data_encoded.drop(["Oral Cancer (Diagnosis)"], axis=1)

##check for class-imbalance
y.mean()
y.sum()
print('Data set comprises {:.0f} positive instances, implying a prevalence of oral cancer of {:.3f}'.format(y.sum(), y.mean()))

# create visualization folder
os.makedirs("data_visualization", exist_ok=True)

# ROC-curve plots
fig_LR, ax_LR = plt.subplots(figsize=(6, 4))
ax_LR.plot([0, 1], [0, 1], linestyle='--', color='red')
ax_LR.set_xlabel("FPR")  
ax_LR.set_ylabel("TPR")  
ax_LR.set_title("ROC Curve - Logistic Regression")  
ax_LR.grid(True)

fig_DT, ax_DT = plt.subplots(figsize=(6, 4))
ax_DT.plot([0, 1], [0, 1], linestyle='--', color='red')
ax_DT.set_xlabel("FPR")  
ax_DT.set_ylabel("TPR")  
ax_DT.set_title("ROC Curve - Decision Tree")  
ax_DT.grid(True)

fig_RF, ax_RF = plt.subplots(figsize=(6, 4))
ax_RF.plot([0, 1], [0, 1], linestyle='--', color='red')
ax_RF.set_xlabel("FPR")  
ax_RF.set_ylabel("TPR")  
ax_RF.set_title("ROC Curve - Random Forest")  
ax_RF.grid(True)

fig_SVM, ax_SVM = plt.subplots(figsize=(6, 4))
ax_SVM.plot([0, 1], [0, 1], linestyle='--', color='red')
ax_SVM.set_xlabel("FPR")  
ax_SVM.set_ylabel("TPR")  
ax_SVM.set_title("ROC Curve - Support Vector Machine") 
ax_SVM.grid(True)


################## CROSS VALIDATION #################

# Perform a 5-fold stratified crossvalidation - prepare the splitting
# HINT: use a class/function from sklearn to do this
# HINT: look at the documentation to find one option that stratifies the splits based on the number of samples in each class
n_splits = 5
skf = StratifiedKFold(n_splits, shuffle = True, random_state = 2025)  # Your solution here 

# Prepare the performance overview data frame - keep this
df_performance = pd.DataFrame(columns = ['fold','clf','accuracy','precision','recall',
                                         'specificity','roc_auc'])
df_LR_normcoef = pd.DataFrame(index = X.columns, columns = np.arange(n_splits))

# Use this counter to save your performance metrics for each crossvalidation fold
# also plot the roc curve for each model and fold into a joint subplot
fold = 0


# Loop over all splits
for train_index, test_index in skf.split(X, y):
    print("Fold:", fold)
    # Get the relevant subsets for training and testing
    X_test  = X.iloc[test_index]  # Your solution here
    y_test  = y.iloc[test_index]  # Your solution here
    X_train = X.iloc[train_index]  # Your solution here
    y_train = y.iloc[train_index]  # Your solution here

    # Standardize the numerical features using training set statistics
    sc = StandardScaler()
    X_train_scaled, X_test_scaled = X_train.copy(), X_test.copy()
    X_train_scaled[num_cols_new] = sc.fit_transform(X_train[num_cols_new])
    X_test_scaled[num_cols_new] = sc.transform(X_test[num_cols_new])

    # Creat prediction models and fit them to the training data
     
    ### Model 1: logistic regression ###
    clf_LR = LogisticRegression(max_iter=1000, random_state=2025)
    clf_LR.fit(X_train_scaled, y_train)
    # Get the importance values - what part of the model do you need here?
    # We provided some skeleton below which should make saving these easier
    df_this_LR_coefs = pd.DataFrame(zip(X_train.columns, np.transpose(clf_LR.coef_[0])), columns=['features', 'coef'])  # Your solution here
    df_LR_normcoef.loc[:,fold] = df_this_LR_coefs['coef'].values/df_this_LR_coefs['coef'].abs().sum()
    # Predictions
    y_test_pred_LR = clf_LR.predict(X_test_scaled)
    y_test_predict_proba_LR = clf_LR.predict_proba(X_test_scaled)[:, 1]
    # confusion matrix and metrics
    cm_LR = confusion_matrix(y_test, y_test_pred_LR)
    tn_LR, fp_LR, fn_LR, tp_LR = cm_LR.ravel()
    # Evaluation metrics
    accuracy_LR = accuracy_score(y_test, y_test_pred_LR)
    precision_LR = precision_score(y_test, y_test_pred_LR)
    recall_LR = recall_score(y_test, y_test_pred_LR)
    specificity_LR = tn_LR / (tn_LR + fp_LR)
    # ROC AUC
    fp_rates_LR, tp_rates_LR, _ = roc_curve(y_test, y_test_predict_proba_LR)
    roc_auc_LR = auc(fp_rates_LR, tp_rates_LR)
    ax_LR.plot(fp_rates_LR, tp_rates_LR, label=f'Logistic Regression (AUC = {roc_auc_LR:.2f})')
    
    ### Model 2: decision tree ###
    clf_DT = tree.DecisionTreeClassifier(random_state=2025)
    clf_DT.fit(X_train, y_train)
    #Predictions
    y_test_pred_DT = clf_DT.predict(X_test)
    y_test_predict_proba_DT = clf_DT.predict_proba(X_test)[:, 1] 
    #confusion matrix
    cm_DT = confusion_matrix(y_test, y_test_pred_DT)
    tn_DT, fp_DT, fn_DT, tp_DT = cm_DT.ravel()
    # Evaluation metrics
    accuracy_DT = accuracy_score(y_test, y_test_pred_DT)
    precision_DT = precision_score(y_test, y_test_pred_DT)
    recall_DT = recall_score(y_test, y_test_pred_DT)
    specificity_DT = tn_DT / (tn_DT + fp_DT)
    # ROC AUC
    # ROC AUC
    fp_rates_DT, tp_rates_DT, _ = roc_curve(y_test, y_test_predict_proba_DT)
    roc_auc_DT = auc(fp_rates_DT, tp_rates_DT)
    ax_DT.plot(fp_rates_DT, tp_rates_DT, label=f'Decision Tree (AUC = {roc_auc_DT:.2f})')
    
    ### Model 3: random forest ###
    # Hyperparameter tuning
    param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10, 20], 'max_features': ['sqrt', 'log2']}
    grid_RF = GridSearchCV(RandomForestClassifier(random_state=2025), param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_RF.fit(X_train, y_train) # No feature scaling for random forest

    # Best model
    clf_RF = grid_RF.best_estimator_

    #clf_RF = RandomForestClassifier(random_state = 2025)  # Only when no hyperparameter tuning
    #clf_RF.fit(X_train, y_train)  # No feature scaling for random forest # Only when no hyperparameter tuning
    # Predictions
    y_test_pred_RF = clf_RF.predict(X_test)
    y_test_predict_proba_RF = clf_RF.predict_proba(X_test)[:, 1] 
    # Confusion matrix
    cm_RF = confusion_matrix(y_test, y_test_pred_RF)
    tn_RF, fp_RF, fn_RF, tp_RF = cm_RF.ravel()
    # Evaluation metrics
    accuracy_RF = accuracy_score(y_test, y_test_pred_RF)
    precision_RF = precision_score(y_test, y_test_pred_RF)
    recall_RF = recall_score(y_test, y_test_pred_RF)
    specificity_RF = tn_RF / (tn_RF + fp_RF)
    # ROC AUC
    fp_rates_RF, tp_rates_RF, _ = roc_curve(y_test, y_test_predict_proba_RF)
    roc_auc_RF = auc(fp_rates_RF, tp_rates_RF)
    ax_RF.plot(fp_rates_RF, tp_rates_RF)
    
    ### Model 4: support vector machine ###
    # Hyperparameter tuning
    svc = SVC(probability=True)
    param_distributions = {'C': uniform(0.1, 100), 'gamma': uniform(0.0001, 1), 'kernel': ['rbf', 'linear']}
    random_search = RandomizedSearchCV(
        estimator=svc,
        param_distributions=param_distributions,
        n_iter=20, # Number of parameter settings to try
        scoring='accuracy', # scoring='roc_auc'
        cv=5,  # inner CV
        verbose=2,
        random_state=42,
        n_jobs=-1 # Use all available cores
    )
    random_search.fit(X_train_scaled, y_train)
    
    # Get best model
    clf_SVM = random_search.best_estimator_

    #clf_SVM = svm.SVC(C=37.55401188473625, gamma=0.9508143064099162, kernel='rbf', probability=True, random_state=42)
    #clf_SVM.fit(X_train_scaled, y_train)
    # Predictions
    y_test_pred_SVM = clf_SVM.predict(X_test_scaled)
    y_test_predict_proba_SVM = clf_SVM.predict_proba(X_test_scaled)[:, 1] 
    # Confusion matrix
    cm_SVM = confusion_matrix(y_test, y_test_pred_SVM)
    tn_SVM, fp_SVM, fn_SVM, tp_SVM = cm_SVM.ravel()
    # Evaluation metrics
    accuracy_SVM = accuracy_score(y_test, y_test_pred_SVM)
    precision_SVM = precision_score(y_test, y_test_pred_SVM)
    recall_SVM = recall_score(y_test, y_test_pred_SVM)
    specificity_SVM = tn_SVM / (tn_SVM + fp_SVM)
    # ROC AUC
    fp_rates_SVM, tp_rates_SVM, _ = roc_curve(y_test, y_test_predict_proba_SVM)
    roc_auc_SVM = auc(fp_rates_SVM, tp_rates_SVM)
    ax_SVM.plot(fp_rates_SVM, tp_rates_SVM, label=f'Support Vector Machine (AUC = {roc_auc_SVM:.2f})')
    
    # Save the performance metrics in the data frame
    df_performance.loc[len(df_performance),:] = [fold, 'LR', accuracy_LR, precision_LR, recall_LR, specificity_LR, roc_auc_LR]
    df_performance.loc[len(df_performance),:] = [fold, 'DT', accuracy_DT, precision_DT, recall_DT, specificity_DT, roc_auc_DT]
    df_performance.loc[len(df_performance), :] = [fold, 'RF', accuracy_RF, precision_RF, recall_RF, specificity_RF, roc_auc_RF]
    df_performance.loc[len(df_performance),:] = [fold, 'SVM', accuracy_SVM, precision_SVM, recall_SVM, specificity_SVM, roc_auc_SVM]
    
    # increase counter for folds
    fold += 1

# Print the results
mean_LR = np.mean(df_performance[df_performance["clf"] == "LR"].iloc[:, 2:], axis = 0)  
std_LR = (df_performance[df_performance["clf"] == "LR"].iloc[:, 2:]).std()  
mean_DT = np.mean(df_performance[df_performance["clf"] == "DT"].iloc[:, 2:], axis = 0)  
std_DT = (df_performance[df_performance["clf"] == "DT"].iloc[:, 2:]).std()  
mean_RF = np.mean(df_performance[df_performance["clf"] == "RF"].iloc[:, 2:], axis = 0)  
std_RF = (df_performance[df_performance["clf"] == "RF"].iloc[:, 2:]).std()  
mean_SVM = np.mean(df_performance[df_performance["clf"] == "SVM"].iloc[:, 2:], axis = 0)  
std_SVM = (df_performance[df_performance["clf"] == "SVM"].iloc[:, 2:]).std()  
mean_std_df_LR_DT = pd.DataFrame({("LR", "mean"): mean_LR, ("LR", "std"): std_LR, ("DT", "mean"): mean_DT, ("DT", "std"): std_DT}) 
mean_std_df_RF_SVM = pd.DataFrame({("RF", "mean"): mean_RF, ("RF", "std"): std_RF, ("SVM", "mean"): mean_SVM, ("SVM", "std"): std_SVM}) 
print()
print(mean_std_df_LR_DT) 
print(mean_std_df_RF_SVM)

# Finishnig and saving the ROC curves
fig_LR.tight_layout()
fig_LR.savefig('roc_curve_LR.png')

fig_DT.tight_layout()
fig_DT.savefig('roc_curve_DT.png')

fig_RF.tight_layout()
fig_RF.savefig('roc_curve_RF.png')

fig_SVM.tight_layout()
fig_SVM.savefig('roc_curve_SVM.png')


