import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt
import seaborn as sns

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



## feature selection with Lasso --> funktioniert nicht, nochmals nachfragen
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




#### evtl. müssen wir noch feature engineering with the interaction terms machen --> wird noch in nachgefragt, ob nötig für unsere research question




#regularisierung mit Lasso




### optional sampling ###






########## Machine Learning Models #########
### Model 1 ###




### Model 2 ###




### Model 3 ###




### Model 4 ###


