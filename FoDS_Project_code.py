import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc)
import os
from sklearn.ensemble import RandomForestClassifier  

from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
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

from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2




from sklearn import tree

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
#print(cate_cols)
data_encoded = pd.get_dummies(data_, prefix=cate_cols, columns=cate_cols, dtype=int)

##construct features and labels
y = data_["Oral Cancer (Diagnosis)"]
X = data_encoded.drop(["Oral Cancer (Diagnosis)"], axis=1)

##check for class-imbalance
y.mean()
y.sum()
print('Data set comprises {:.0f} positive instances, implying a prevalence of oral cancer of {:.3f}'.format(y.sum(), y.mean()))

##train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2025)
X_train_unscaled = X_train

##standardization
sc = MinMaxScaler() #StandardScaler()
num_cols_new = num_cols.drop(["Oral Cancer (Diagnosis)"])

X_train_scaled, X_test_scaled = X_train.copy(), X_test.copy()
X_train_scaled[num_cols_new] = sc.fit_transform(X_train[num_cols_new])
X_test_scaled[num_cols_new] = sc.transform(X_test[num_cols_new])

## Convert pandas DataFrame to numpy array
X_train, X_test, y_train, y_test = (
    np.array(X_train),
    np.array(X_test),
    np.array(y_train),
    np.array(y_test),
)
"""
## feature selection with Lasso
lasso_cv = LassoCV(cv=5, random_state=0)
lasso_cv.fit(X_train_scaled, y_train)

print("Optimal alpha:", lasso_cv.alpha_)
print("Coefficients:", lasso_cv.coef_)

selected_mask = lasso_cv.coef_ != 0
selected_features = X_train_scaled.columns[selected_mask]
X_train_selected = X_train_scaled[selected_features]
X_test_selected = X_test_scaled[selected_features]
print(selected_features)
"""
## feature selection with Univariate FS
UVFS_Selector = SelectKBest(chi2, k=5) # Select top 4 features
X_UVFS = UVFS_Selector.fit_transform(X_train_scaled, y_train) # ...but only on training data.
X_UVFS_test = UVFS_Selector.transform(X_test)
scores = -np.log10(UVFS_Selector.pvalues_)
scores /= scores.max()

# Plot 
X_indices = np.arange(X.shape[-1])
plt.figure()
plt.clf()
plt.bar(X_indices - 0.05, scores, width=0.2)
plt.title("Feature univariate score")
plt.xlabel("Feature")
plt.ylabel(r"Univariate score ($-Log(p_{value})$)")
plt.xticks(X_indices, X.columns, rotation = 90)
plt.tight_layout()
plt.savefig("Univariate score of features")


mask = UVFS_Selector.get_support()
all_feature_names = X.columns
p_values = UVFS_Selector.pvalues_
selected_features = [(name, pval) for name, pval, selected in zip(all_feature_names, p_values, mask) if selected]
# Print features and their p-values
print("Selected Features and Their p-values (Top 5 by chi-squared test):")
for name, pval in selected_features:
    print(f"{name}: p-value = {pval:.4e}")
#constructed selected X_train
X_train_selected = X_train_unscaled.loc[:, mask]
print(X_train_selected)















### Data visualization ###
plt.figure(figsize = (8,6))
age_before = sns.histplot(data = X_train_unscaled, x = "Age", bins = 20)
age_before.set_xlabel("Age")
age_before.set_ylabel("Count")
age_before.set_title("Age distribution before data preprocessing")
plt.savefig("age_before_preprocessing.jpg")

plt.figure(figsize = (8,6))
age_after = sns.histplot(data = X_train_scaled, x = "Age", bins = 20)
age_after.set_xlabel("Age")
age_after.set_ylabel("Count")
age_after.set_title("Age distribution after data preprocessing")
plt.savefig("age_after_preprocessing.jpg")


print("")
print("")
########## Machine Learning Models #########
#regularisierung mit Lasso
### Model 1: logistic regression ###

# train model
log_reg = LogisticRegression(max_iter=1000, random_state=2025)
log_reg.fit(X_train_scaled, y_train)

# predictions
y_test_pred_log = log_reg.predict(X_test_scaled)
y_test_predict_proba_log = log_reg.predict_proba(X_test_scaled)[:, 1]

# confusion matrix and metrics
cm_log = confusion_matrix(y_test, y_test_pred_log)
tn_log, fp_log, fn_log, tp_log = cm_log.ravel()
specificity_log = tn_log / (tn_log + fp_log)

print("Confusion Matrix (Logistic Regression):\n", cm_log)
print(f"Accuracy: {accuracy_score(y_test, y_test_pred_log):.3f}")
print(f"Precision: {precision_score(y_test, y_test_pred_log):.3f}")
print(f"Recall: {recall_score(y_test, y_test_pred_log):.3f}")
print(f"Specificity: {specificity_log:.3f}")
print(f"F1 Score: {f1_score(y_test, y_test_pred_log):.3f}")

# ROC AUC
fpr_log, tpr_log, _ = roc_curve(y_test, y_test_predict_proba_log)
roc_auc_log = auc(fpr_log, tpr_log)
print(f"ROC AUC: {roc_auc_log:.3f}")

# create visualization folder
os.makedirs("data_visualization", exist_ok=True)

# ROC Curve
plt.figure(figsize=(6, 4))
plt.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {roc_auc_log:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='red')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve - Logistic Regression')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("data_visualization/roc_curve_LR.jpg")

# 1. Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm_log, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("data_visualization/confusion_matrix_logistic_regression.jpg")
plt.close()

# 2. Feature Importance (Top 10 Coefficients)
# Only possible if X_train_scaled is still a DataFrame
if hasattr(X_train_scaled, 'columns'):
    feature_names = X_train_scaled.columns
    coef = pd.Series(log_reg.coef_[0], index=feature_names)
    top_features = coef.abs().sort_values(ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    top_features.sort_values().plot(kind="barh")
    plt.title("Top 10 Influential Features - Logistic Regression")
    plt.xlabel("Coefficient Magnitude")
    plt.tight_layout()
    plt.savefig("data_visualization/logistic_regression_top_features.jpg")
    plt.close()



print("")
print("")
### Model 2: decision tree ###
print("######### DECISION TREE #########")
dt = tree.DecisionTreeClassifier(random_state=2025)
dt.fit(X_train_selected, y_train)

#Predictions
y_test_pred = dt.predict(X_test_selected)
y_test_predict_proba = dt.predict_proba(X_test_selected)[:, 1] 

#confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
print("Confusion matrix:\n", cm)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)

# ROC AUC
fp_rates, tp_rates, _ = roc_curve(y_test, y_test_predict_proba)
roc_auc = auc(fp_rates, tp_rates)

# ROC curve
plt.figure(figsize = (6, 4))
plt.plot(fp_rates, tp_rates, label=f'Decision Tree (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='red')
plt.xlabel("FPR")  
plt.ylabel("TPR")  
plt.title("ROC Curve - Decision Tree")  
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('data_visualization/roc_curve_DT.png')

# Print the results
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"ROC AUC: {roc_auc:.3f}")

#plot decision tree
plt.figure(figsize=(20,10))
tree.plot_tree(dt, filled=True)
plt.savefig('decision tree.png')


print("")
print("")
print("######### DECISION TREE (NO PRESELECTION) #########")
t = tree.DecisionTreeClassifier(random_state=2025)
dt.fit(X_train, y_train)

#Predictions
y_test_pred = dt.predict(X_test)
y_test_predict_proba = dt.predict_proba(X_test)[:, 1] 

#confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
print("Confusion matrix:\n", cm)

#plot decision tree
plt.figure(figsize=(20,10))
tree.plot_tree(dt, filled=True)
plt.savefig('decision tree.png')


print("")
print("")
print("### Model 3: random forest ###")
rf = RandomForestClassifier(random_state = 2025)
rf.fit(X_train, y_train)

# Predictions
y_test_pred = rf.predict(X_test)
y_test_predict_proba = rf.predict_proba(X_test)[:, 1] 

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
print("Confusion matrix:\n", cm)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
# Specificity (TN / (TN + FP)) 
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)

# ROC AUC
fp_rates, tp_rates, _ = roc_curve(y_test, y_test_predict_proba)
roc_auc = auc(fp_rates, tp_rates)

# ROC curve
plt.figure(figsize = (6, 4))
plt.plot(fp_rates, tp_rates, label=f'Random Forest (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='red')
plt.xlabel("FPR")  
plt.ylabel("TPR")  
plt.title("ROC Curve - Random Forest")  
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('data_visualization/roc_curve_RF.png')

# Print the results
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"ROC AUC: {roc_auc:.3f}")




print("")
print("")
print("### Model 4: support vector machine ###")
print("## Hyperparameter tuning for SVM ##")

svc = SVC()

param_distributions = {'C': uniform(0.1, 100),'gamma': uniform(0.0001, 1),'kernel': ['rbf', 'linear']}
random_search = RandomizedSearchCV(
    estimator=svc,
    param_distributions=param_distributions,
    n_iter=20,          # Number of parameter settings to try
    scoring='accuracy', 
    cv=5,               
    verbose=2,
    random_state=42,
    n_jobs=-1           # Use all available cores
)

random_search.fit(X_train_selected, y_train)

print("Best parameters found: ", random_search.best_params_) #Best parameters found:  {'C': np.float64(37.55401188473625), 'gamma': np.float64(0.9508143064099162), 'kernel': 'rbf'}

print("")
print("## SVM Model ##")
## Support Vector Machine Model ##
#function for evaluation metrics
clf_SVM = svm.SVC(
    C=37.55401188473625,
    gamma=0.9508143064099162,
    kernel='rbf',
    probability=True,
    random_state=42)

clf_SVM.fit(X_train_selected, y_train)

# Predictions
y_test_pred = clf_SVM.predict(X_test_selected)
y_test_predict_proba = clf_SVM.predict_proba(X_test_selected)[:, 1] 

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
print("Confusion matrix: (Support Vector Machine)\n", cm)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)

# ROC AUC
fp_rates, tp_rates, _ = roc_curve(y_test, y_test_predict_proba)
roc_auc = auc(fp_rates, tp_rates)

# ROC curve
plt.figure(figsize = (6, 4))
plt.plot(fp_rates, tp_rates, label=f'Support Vector Machine = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='red')
plt.xlabel("FPR")  
plt.ylabel("TPR")  
plt.title("ROC Curve - Support Vector Machine")  
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('data_visualization/roc_curve_SVM.png')

# Print the results
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"ROC AUC: {roc_auc:.3f}")
