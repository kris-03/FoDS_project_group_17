import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold  
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, auc)
import os
from sklearn.ensemble import RandomForestClassifier  
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
import warnings
warnings.filterwarnings("ignore")


# Loading the data
data = pd.read_csv(
    filepath_or_buffer = "oral_cancer_prediction_dataset.csv",
    dtype = {
        "Country": "category", "Gender": "category", "Tobacco Use": "category", "Alcohol Consumption": "category", "HPV Infection": "category",
        "Betel Quid Use": "category", "Chronic Sun Exposure": "category", "Poor Oral Hygiene": "category", "Diet (Fruits & Vegetables Intake)": "category",
        "Family History of Cancer": "category", "Compromised Immune System": "category", "Oral Lesions": "category", "Unexplained Bleeding": "category",
        "Difficulty Swallowing": "category", "White or Red Patches in Mouth": "category", "Cancer Stage": "category",  "Treatment Type": "category",
        "Early Diagnosis": "category"
    }
)

######### Methods ###########
# Missing values
print("Number of missing values:", data.isna().sum(axis = 1).sum()) #output is 0, so we have no missing date --> no special handling needed

# Changing label oral cancer from true/false to 1/0
data["Oral Cancer (Diagnosis)"] = data["Oral Cancer (Diagnosis)"].map({"No": 0, "Yes": 1})

# Dropping all columns giving a hint to the outcome (and also the column ID)
data_ = data.drop(["ID", "Early Diagnosis", "Treatment Type", "Cancer Stage", "Survival Rate (5-Year, %)", "Cost of Treatment (USD)", "Economic Burden (Lost Workdays per Year)" , "Tumor Size (cm)"], axis=1)
#print(data_.columns)

# One-hot-encoding
cate_cols = data_.columns[data_.dtypes == "category"]
num_cols = data_.columns[data_.dtypes != "category"]
num_cols_new = num_cols.drop(["Oral Cancer (Diagnosis)"])
data_encoded = pd.get_dummies(data_, prefix=cate_cols, columns=cate_cols, dtype=int)

# Construct features and labels
y = data_["Oral Cancer (Diagnosis)"]
X = data_encoded.drop(["Oral Cancer (Diagnosis)"], axis=1)

# Check for class-imbalance
y.mean()
y.sum()
print('Data set comprises {:.0f} positive instances, implying a prevalence of oral cancer of {:.3f}'.format(y.sum(), y.mean()))

# Create visualization folder
os.makedirs("data_visualization", exist_ok = True)

# ROC-curve plots
fig_LR, ax_LR = plt.subplots(figsize = (6, 4))
ax_LR.plot([0, 1], [0, 1], linestyle = '--', color='red')
ax_LR.set_xlabel("FPR")  
ax_LR.set_ylabel("TPR")  
ax_LR.set_title("ROC Curve - Logistic Regression with feature selection")  
ax_LR.grid(True)

fig_DT, ax_DT = plt.subplots(figsize = (6, 4))
ax_DT.plot([0, 1], [0, 1], linestyle = '--', color='red')
ax_DT.set_xlabel("FPR")  
ax_DT.set_ylabel("TPR")  
ax_DT.set_title("ROC Curve - Decision Tree with feature selection")  
ax_DT.grid(True)

fig_RF, ax_RF = plt.subplots(figsize = (6, 4))
ax_RF.plot([0, 1], [0, 1], linestyle = '--', color='red')
ax_RF.set_xlabel("FPR")  
ax_RF.set_ylabel("TPR")  
ax_RF.set_title("ROC Curve - Random Forest with feature selection")  
ax_RF.grid(True)

fig_SVM, ax_SVM = plt.subplots(figsize = (6, 4))
ax_SVM.plot([0, 1], [0, 1], linestyle = '--', color='red')
ax_SVM.set_xlabel("FPR")  
ax_SVM.set_ylabel("TPR")  
ax_SVM.set_title("ROC Curve - Support Vector Machine with feature selection") 
ax_SVM.grid(True)


################## CROSS VALIDATION #################

# Perform a 5-fold stratified crossvalidation 
n_splits = 5
skf = StratifiedKFold(n_splits, shuffle = True, random_state = 42)  

# Performance overview data frame 
df_performance = pd.DataFrame(columns = ['fold','clf','accuracy','precision','recall', 'specificity','roc_auc'])
df_performance_all = pd.DataFrame(columns = ['fold','clf','accuracy','precision','recall', 'specificity','roc_auc'])

# Feature importance data frame
feature_coefficient_LR = []
df_feature_importance_DT = pd.DataFrame(index = X.columns)
df_feature_importance_RF = pd.DataFrame(index = X.columns)


fold = 0 # Counter

# Loop over all splits
for train_index, test_index in skf.split(X, y):
    print("Fold:", fold)
    # Get the relevant subsets for training and testing
    X_test  = X.iloc[test_index]  
    y_test  = y.iloc[test_index] 
    X_train = X.iloc[train_index]  
    y_train = y.iloc[train_index]  
    
    # Standardization
    sc = MinMaxScaler() 
    X_train_scaled, X_test_scaled = X_train.copy(), X_test.copy()
    X_train_scaled[num_cols_new] = sc.fit_transform(X_train[num_cols_new])
    X_test_scaled[num_cols_new] = sc.transform(X_test[num_cols_new])

    # Feature selection with Univariate FS
    UVFS_Selector = SelectKBest(chi2, k = 5) # Select top 5 features
    X_UVFS = UVFS_Selector.fit_transform(X_train_scaled, y_train)
    X_UVFS_test = UVFS_Selector.transform(X_test)
    scores = -np.log10(UVFS_Selector.pvalues_)
    scores /= scores.max()

    # Plot 
    X_indices = np.arange(X.shape[-1])
    plt.figure()
    plt.clf()
    plt.bar(X_indices - 0.05, scores, width = 0.2)
    plt.title(f"Feature univariate score (Fold {fold})")
    plt.xlabel("Feature")
    plt.ylabel(r"Univariate score ($-Log(p_{value})$)")
    plt.xticks(X_indices, X.columns, rotation = 90)
    plt.tight_layout()
    plt.savefig(f"data_visualization/UVFS_scores_fold_{fold}.png")

    mask = UVFS_Selector.get_support()
    all_feature_names = X.columns
    p_values = UVFS_Selector.pvalues_
    selected_features = [(name, pval) for name, pval, selected in zip(all_feature_names, p_values, mask) if selected]
    # Print features and their p-values
    print("Selected Features and Their p-values (Top 5 by chi-squared test):")
    for name, pval in selected_features:
        print(f"{name}: p-value = {pval:.4e}")
    
    # Constructed selected X_train
    X_train_selected = X_train.loc[:, mask] # With unscaled data for DT and RF
    X_test_selected = X_test.loc[:, mask]
    X_train_scaled_selected = X_train_scaled.loc[:, mask] # With scaled data for LR and SVM
    X_test_scaled_selected = X_test_scaled.loc[:, mask]
    
    # Creat prediction models and fit them to the training data (with feature selection)
     
    ### Model 1: logistic regression ###
    clf_LR = LogisticRegression(max_iter = 1000, random_state = 42)
    clf_LR_all = LogisticRegression(max_iter = 1000, random_state = 42)

    # Logistic regression with feature selection
    clf_LR.fit(X_train_scaled_selected, y_train)
    # Predictions
    y_test_pred_LR = clf_LR.predict(X_test_scaled_selected)
    y_test_predict_proba_LR = clf_LR.predict_proba(X_test_scaled_selected)[:, 1]
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
    ax_LR.plot(fp_rates_LR, tp_rates_LR, label = f'Fold {fold} (AUC = {roc_auc_LR:.2f})')

    
    # Logistic regression without feature selection
    clf_LR_all.fit(X_train_scaled, y_train)
    # Predictions
    y_test_pred_LR_all = clf_LR_all.predict(X_test_scaled)
    y_test_predict_proba_LR_all = clf_LR_all.predict_proba(X_test_scaled)[:, 1]
    # confusion matrix and metrics
    cm_LR_all = confusion_matrix(y_test, y_test_pred_LR_all)
    tn_LR_all, fp_LR_all, fn_LR_all, tp_LR_all = cm_LR_all.ravel()
    # Evaluation metrics
    accuracy_LR_all = accuracy_score(y_test, y_test_pred_LR_all)
    precision_LR_all = precision_score(y_test, y_test_pred_LR_all)
    recall_LR_all = recall_score(y_test, y_test_pred_LR_all)
    specificity_LR_all = tn_LR_all / (tn_LR_all + fp_LR_all)
    # ROC AUC
    fp_rates_LR_all, tp_rates_LR_all, _ = roc_curve(y_test, y_test_predict_proba_LR_all)
    roc_auc_LR_all = auc(fp_rates_LR_all, tp_rates_LR_all)
    # Feature coefficients
    df_this_LR_coef = pd.DataFrame({'feature': X_train_scaled.columns, 'coef': clf_LR_all.coef_[0]})
    df_this_LR_coef['fold'] = fold
    feature_coefficient_LR.append(df_this_LR_coef)   
    
    # Save performance metrics
    df_performance.loc[len(df_performance),:] = [fold, 'LR', accuracy_LR, precision_LR, recall_LR, specificity_LR, roc_auc_LR]
    df_performance_all.loc[len(df_performance_all),:] = [fold, 'LR', accuracy_LR_all, precision_LR_all, recall_LR_all, specificity_LR_all, roc_auc_LR_all]

    ### Model 2: decision tree ###
    clf_DT = tree.DecisionTreeClassifier(random_state = 42)
    clf_DT_all = tree.DecisionTreeClassifier(random_state = 42)
    
    # Decision Tree with feature selection
    clf_DT.fit(X_train_selected, y_train)
    #Predictions
    y_test_pred_DT = clf_DT.predict(X_test_selected)
    y_test_predict_proba_DT = clf_DT.predict_proba(X_test_selected)[:, 1] 
    #confusion matrix
    cm_DT = confusion_matrix(y_test, y_test_pred_DT)
    tn_DT, fp_DT, fn_DT, tp_DT = cm_DT.ravel()
    # Evaluation metrics
    accuracy_DT = accuracy_score(y_test, y_test_pred_DT)
    precision_DT = precision_score(y_test, y_test_pred_DT)
    recall_DT = recall_score(y_test, y_test_pred_DT)
    specificity_DT = tn_DT / (tn_DT + fp_DT)
    # ROC AUC
    fp_rates_DT, tp_rates_DT, _ = roc_curve(y_test, y_test_predict_proba_DT)
    roc_auc_DT = auc(fp_rates_DT, tp_rates_DT)
    ax_DT.plot(fp_rates_DT, tp_rates_DT, label = f'Fold {fold} (AUC = {roc_auc_DT:.2f})')

    # Decision Tree without feature selection
    clf_DT_all.fit(X_train, y_train)
    # Predictions
    y_test_pred_DT_all = clf_DT_all.predict(X_test)
    y_test_predict_proba_DT_all = clf_DT_all.predict_proba(X_test)[:, 1] 
    # Confusion matrix
    cm_DT_all = confusion_matrix(y_test, y_test_pred_DT_all)
    tn_DT_all, fp_DT_all, fn_DT_all, tp_DT_all = cm_DT_all.ravel()
    # Evaluation metrics
    accuracy_DT_all = accuracy_score(y_test, y_test_pred_DT_all)
    precision_DT_all = precision_score(y_test, y_test_pred_DT_all)
    recall_DT_all = recall_score(y_test, y_test_pred_DT_all)
    specificity_DT_all = tn_DT_all / (tn_DT_all + fp_DT_all)
    # ROC AUC
    fp_rates_DT_all, tp_rates_DT_all, _ = roc_curve(y_test, y_test_predict_proba_DT_all)
    roc_auc_DT_all = auc(fp_rates_DT_all, tp_rates_DT_all)
    # Feature importances
    importances = clf_DT_all.feature_importances_
    df_feature_importance_DT[f'fold_{fold}'] = importances
    
    # Save performance metrics
    df_performance.loc[len(df_performance), :] = [fold, 'DT', accuracy_DT, precision_DT, recall_DT, specificity_DT, roc_auc_DT]
    df_performance_all.loc[len(df_performance_all), :] = [fold, 'DT', accuracy_DT_all, precision_DT_all, recall_DT_all, specificity_DT_all, roc_auc_DT_all]

    ### Model 3: random forest ###
    # Hyperparameters
    param_grid = {'n_estimators': [50, 100, 200], 
                  'max_depth': [None, 2, 5, 10, 20], 
                  'max_features': ['sqrt', 'log2', None],
                  'min_samples_split': [2, 5],
                  'min_samples_leaf': [1, 2],
                  'bootstrap': [True, False], 
                  'criterion': ['gini', 'entropy']
                  }
    grid_RF = GridSearchCV(RandomForestClassifier(random_state = 42), param_grid, cv = 3, scoring = 'accuracy', n_jobs = -1)
    grid_RF_all = GridSearchCV(RandomForestClassifier(random_state = 42), param_grid, cv = 3, scoring = 'accuracy', n_jobs = -1)
    
    # Random forest with feature selection
    grid_RF.fit(X_train_selected, y_train) # No feature scaling for random forest
    print("Best Parameters for Random Forest with feature selection:", grid_RF.best_params_)
    # Best model
    clf_RF = grid_RF.best_estimator_
    # Predictions
    y_test_pred_RF = clf_RF.predict(X_test_selected)
    y_test_predict_proba_RF = clf_RF.predict_proba(X_test_selected)[:, 1] 
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
    ax_RF.plot(fp_rates_RF, tp_rates_RF, label = f'Fold {fold} (AUC = {roc_auc_RF:.2f})')    

    # Random forest without feature selection
    grid_RF_all.fit(X_train, y_train) # No feature scaling for random forest
    print("Best Parameters for Random Forest without feature selection:", grid_RF_all.best_params_)
    clf_RF_all = grid_RF_all.best_estimator_
    # Predictions
    y_test_pred_RF_all = clf_RF_all.predict(X_test)
    y_test_predict_proba_RF_all = clf_RF_all.predict_proba(X_test)[:, 1] 
    # Confusion matrix
    cm_RF_all = confusion_matrix(y_test, y_test_pred_RF_all)
    tn_RF_all, fp_RF_all, fn_RF_all, tp_RF_all = cm_RF_all.ravel()
    # Evaluation metrics
    accuracy_RF_all = accuracy_score(y_test, y_test_pred_RF_all)
    precision_RF_all = precision_score(y_test, y_test_pred_RF_all)
    recall_RF_all = recall_score(y_test, y_test_pred_RF_all)
    specificity_RF_all = tn_RF_all / (tn_RF_all + fp_RF_all)
    # ROC AUC
    fp_rates_RF_all, tp_rates_RF_all, _ = roc_curve(y_test, y_test_predict_proba_RF_all)
    roc_auc_RF_all = auc(fp_rates_RF_all, tp_rates_RF_all)
    # Feature importances
    importances = clf_RF_all.feature_importances_
    df_feature_importance_RF[f'fold_{fold}'] = importances
    
    # Save performance metrics
    df_performance.loc[len(df_performance), :] = [fold, 'RF', accuracy_RF, precision_RF, recall_RF, specificity_RF, roc_auc_RF]
    df_performance_all.loc[len(df_performance_all), :] = [fold, 'RF', accuracy_RF_all, precision_RF_all, recall_RF_all, specificity_RF_all, roc_auc_RF_all]

    ### Model 4: support vector machine ###
    # Hyperparameter tuning
    svc = SVC(probability = True, random_state = 42)
    param_distributions = [
        {'kernel': ['linear'], 'C': uniform(0.1, 100)},
        {'kernel': ['rbf'], 'C': uniform(0.1, 100), 'gamma': uniform(0.0001, 1)}]
    random_search = RandomizedSearchCV(
        estimator = svc,
        param_distributions = param_distributions,
        n_iter = 10, # Number of parameter settings to try
        scoring = 'accuracy', 
        cv = 3,  # Inner CV
        verbose = 2,
        random_state = 42,
        n_jobs = -1 # Use all available cores
    )
    random_search.fit(X_train_scaled_selected, y_train)
    print("Best parameters for Support Vector Machine: ", random_search.best_params_) 
    
    # Best model
    clf_SVM = random_search.best_estimator_
    # Predictions
    y_test_pred_SVM = clf_SVM.predict(X_test_scaled_selected)
    y_test_predict_proba_SVM = clf_SVM.predict_proba(X_test_scaled_selected)[:, 1] 
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
    ax_SVM.plot(fp_rates_SVM, tp_rates_SVM, label=f'Fold {fold} (AUC = {roc_auc_SVM:.2f})')
    
    # Save performance metrics
    df_performance.loc[len(df_performance),:] = [fold, 'SVM', accuracy_SVM, precision_SVM, recall_SVM, specificity_SVM, roc_auc_SVM]
    
    # increase counter for folds
    fold += 1

# Print the results without feature selection
print("Performance with feature selection")
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

# Print the results wit feature selection
print("Performance without feature selection:")
mean_LR_all = np.mean(df_performance_all[df_performance_all["clf"] == "LR"].iloc[:, 2:], axis = 0)  
std_LR_all = (df_performance_all[df_performance_all["clf"] == "LR"].iloc[:, 2:]).std()  
mean_DT_all = np.mean(df_performance_all[df_performance_all["clf"] == "DT"].iloc[:, 2:], axis = 0)  
std_DT_all = (df_performance_all[df_performance_all["clf"] == "DT"].iloc[:, 2:]).std()  
mean_RF_all = np.mean(df_performance_all[df_performance_all["clf"] == "RF"].iloc[:, 2:], axis = 0)  
std_RF_all = (df_performance_all[df_performance_all["clf"] == "RF"].iloc[:, 2:]).std()  
mean_std_df_LR_DT_all = pd.DataFrame({("LR", "mean"): mean_LR_all, ("LR", "std"): std_LR_all, ("DT", "mean"): mean_DT_all, ("DT", "std"): std_DT_all}) 
mean_std_df_RF_all = pd.DataFrame({("RF", "mean"): mean_RF_all, ("RF", "std"): std_RF_all}) 
print()
print(mean_std_df_LR_DT_all) 
print(mean_std_df_RF_all)

# Finishnig and saving the ROC curves
ax_LR.legend()
fig_LR.tight_layout()
fig_LR.savefig('data_visualization/roc_curve_LR.png')

ax_DT.legend()
fig_DT.tight_layout()
fig_DT.savefig('data_visualization/roc_curve_DT.png')

ax_RF.legend()
fig_RF.tight_layout()
fig_RF.savefig('data_visualization/roc_curve_RF.png')

ax_SVM.legend()
fig_SVM.tight_layout()
fig_SVM.savefig('data_visualization/roc_curve_SVM.png')

# Plot of top 10 most influential features

# Mean coefficients / importances across folds
df_feature_coefficient_LR_folds = pd.concat(feature_coefficient_LR, axis = 0)
df_feature_coefficient_LR = df_feature_coefficient_LR_folds.groupby('feature')['coef'].agg(['mean', 'std'])
df_feature_coefficient_LR['abs_mean'] = df_feature_coefficient_LR['mean'].abs()
df_feature_importance_DT['mean_importance'] = df_feature_importance_DT.mean(axis = 1)
df_feature_importance_RF['mean_importance'] = df_feature_importance_RF.mean(axis = 1)

# Sort and select top 10 features
top10 = df_feature_coefficient_LR.sort_values('abs_mean', ascending = False).head(10)
top_features = df_feature_importance_DT['mean_importance'].sort_values(ascending = False).head(10)
top_features = df_feature_importance_RF['mean_importance'].sort_values(ascending = False).head(10)

# Plotting
plt.figure(figsize = (10, 6))
top10['mean'].sort_values().plot(kind = 'barh')
plt.title('Top 10 Influential Features - Logistic Regression')
plt.xlabel('Mean Coefficient')
plt.tight_layout()
plt.savefig('data_visualization/LR_feature_coefficients.jpg')

plt.figure(figsize = (10, 6))
top_features.sort_values().plot(kind = 'barh')
plt.title('Top 10 Influential Features - Decision Tree')
plt.xlabel('Mean Feature Importance')
plt.tight_layout()
plt.savefig('data_visualization/DT_feature_importance.jpg')

plt.figure(figsize = (10, 6))
top_features.sort_values().plot(kind = 'barh')
plt.title('Top 10 Influential Features - Random Forest')
plt.xlabel('Mean Feature Importance')
plt.tight_layout()
plt.savefig('data_visualization/RF_feature_importance.jpg')
