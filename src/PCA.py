import pandas as pd
from src.data_management import my_imputer, prep_data, stay_num, drop, mark_missing, dummify
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from dabl.search import GridSuccessiveHalving
import time
from sklearn.decomposition import PCA

# Training dataset
data = pd.read_csv('../data/mimic_train.csv')
kaggle = pd.read_csv('../data/mimic_test_los.csv')

data_join = data.append(kaggle)

data_join = data_join.reset_index(drop=True)

data_join = prep_data(data_join)
data_join = stay_num(data_join)
data_join = mark_missing(data_join, ['HOSPITAL_EXPIRE_FLAG', 'LOS', 'MY_LOS'])
data_join = my_imputer(data_join)

# Cathegorical Variables
cathegorical_vars = ['ADMISSION_TYPE', 'INSURANCE', 'FIRST_CAREUNIT', 'ETHNICITY', 'GENDER', 'MARITAL_STATUS', 'ICD9_diagnosis']
data_join_2 = dummify(data_join, cathegorical_vars)

data_2 = data_join_2[:len(data)]
kaggle_2 = data_join_2[len(data):]

my_columns = data_2.select_dtypes(include=['float64']).columns

pca_columns = list(set(my_columns) - set(['HOSPITAL_EXPIRE_FLAG', 'LOS', 'MY_ADMIN_TIME',
       'MY_ADMIN_WEEKDAY_DEC', 'MY_ADMIN_MONTH_COS', 'MY_ADMIN_MONTH_SIN',
       'MY_CHECKOUT_TIME', 'MY_CHECKOUT_WEEKDAY', 'MY_CHECKOUT_WEEKDAY_DEC',
       'MY_LOS', 'LOS_DEC']))

pca_data = data_2[pca_columns]

pca = PCA().fit(pca_data)

post_pca = data_2.drop(pca_columns, axis=1)

aux = pca.transform(pca_data)

post_pca[['F_1', 'F_2', 'F_3', 'F_4', 'F_5']] = aux[:, :5]


print('hola')