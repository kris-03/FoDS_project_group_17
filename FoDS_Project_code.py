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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm


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
print(data.isna().sum(axis=1).sum()) #output is 0, so we have no missing date --> no special handling needed

##changing label oral cancer from true/false to 1/0
data["Oral Cancer (Diagnosis)"] = data["Oral Cancer (Diagnosis)"].map({"No": 0, "Yes": 1})

##first dropping all columns giving a hint to the outcome
data_ = data.drop(["Early Diagnosis", "Treatment Type", "Cancer Stage", "Survival Rate (5-Year, %)", "Cost of Treatment (USD)", "Economic Burden (Lost Workdays per Year)" , "Tumor Size (cm)"], axis=1)
#print(data_.columns)

##one-hot-encoding
cate_cols = data_.columns[data_.dtypes == "category"]
num_cols = data_.columns[data_.dtypes != "category"]
#print(cate_cols)
data_encoded = pd.get_dummies(data_, prefix=cate_cols, columns=cate_cols, dtype=int)

##construct features and labels
y = data_["Oral Cancer (Diagnosis)"]
X = data_encoded.drop(["Oral Cancer (Diagnosis)"], axis=1)

##train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2025)
X_train_unscaled = X_train

##standardization
sc = StandardScaler()
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

## feature selection with Lasso --> alles 0, weil kein feature wichtig ist
def get_scores(model, X_train, y_train, X_test, y_test):
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    # evaluation
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = root_mean_squared_error(y_test, y_pred_test)

    r2_train = r2_score(y_train, y_pred_train)
    rmse_train = root_mean_squared_error(y_train, y_pred_train)

    print('Training set score: R2 score: {:.3f}, RMSE: {:.3f}'.format(r2_train, rmse_train))
    print('Test set score: R2 score: {:.3f}, RMSE: {:.3f}'.format(r2_test, rmse_test))

Lasso = Lasso(alpha=0.1)
Lasso.fit(X_train_scaled, y_train) 
# get scores
get_scores(Lasso, X_train_scaled, y_train, X_test_scaled, y_test)
print(Lasso.coef_)

### optional sampling ###

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
plt.figure(figsize=(8,6))
plt.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {roc_auc_log:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("data_visualization/roc_logistic_regression.jpg")
plt.close()

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



### Model 2: decision tree ###




### Model 3: random forest ###
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
plt.figure(figsize = (9, 6))
plt.plot(fp_rates, tp_rates, label=f'ROC curve AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='red')
plt.xlabel("FPR")  
plt.ylabel("TPR")  
plt.title("ROC curve for Random Forest model")  # Your solution here
plt.tight_layout()
plt.savefig('roc_curve_RF.png')

# Print the results
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"ROC AUC: {roc_auc:.3f}")

### Model 4: support vector machine ###
#function for evaluation metrics
clf_SVM = svm.SVC(probability=True, kernel = 'linear')
clf_SVM.fit(X_train, y_train)

# Predictions
y_test_pred = clf_SVM.predict(X_test)
y_test_predict_proba = clf_SVM.predict_proba(X_test)[:, 1] 

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
plt.figure(figsize = (9, 6))
plt.plot(fp_rates, tp_rates, label=f'ROC curve AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='red')
plt.xlabel("FPR")  
plt.ylabel("TPR")  
plt.title("ROC curve for Support Vector Machine")  
plt.tight_layout()
plt.savefig('roc_curve_SVM.png')

# Print the results
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"ROC AUC: {roc_auc:.3f}")
