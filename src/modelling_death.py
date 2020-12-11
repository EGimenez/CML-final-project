import pandas as pd
from src.data_management import my_imputer, prep_data, stay_num, drop, mark_missing, dummify, my_pca
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from dabl.search import GridSuccessiveHalving
import time

# Training dataset
data = pd.read_csv('../data/mimic_train.csv')
kaggle = pd.read_csv('../data/mimic_test_los.csv')

data_join = data.append(kaggle)

data_join = data_join.reset_index(drop=True)

data_join = prep_data(data_join)
data_join = stay_num(data_join)
data_join = mark_missing(data_join, ['HOSPITAL_EXPIRE_FLAG', 'LOS', 'MY_LOS'])
data_join = my_imputer(data_join)
#data_join = my_pca(data_join)

# Cathegorical Variables
cathegorical_vars = ['ADMISSION_TYPE', 'INSURANCE', 'FIRST_CAREUNIT', 'ETHNICITY', 'GENDER', 'MARITAL_STATUS', 'ICD9_diagnosis']
data_join_2 = dummify(data_join, cathegorical_vars)

data_2 = data_join_2[:len(data)]
kaggle_2 = data_join_2[len(data):]


# Caracterizamos Death
data_X = drop(data_2, ['hadm_id', 'icustay_id', 'SPECIAL_ADMIN', 'MY_CHECKOUT_DATE', 'MY_CHECKOUT_TIME',
                       'MY_CHECKOUT_WEEKDAY', 'MY_CHECKOUT_WEEKDAY_DEC', 'SPECIAL_ADMIN', 'MY_LOS',
                       'LOS_DEC', 'LOS', 'HOSPITAL_EXPIRE_FLAG', 'MY_ADMIN_DATE'],
              is_reg=True)

data_Y = data_2['HOSPITAL_EXPIRE_FLAG']

params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5],
        'n_estimators': range(50, 400, 50)
        }

params = {
        'max_depth': [5, 6],
        'n_estimators': range(400, 700, 50)
        }


xgb = xgb.XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                    silent=True, nthread=1)

# Halving

t0 = time.time()
sh = GridSuccessiveHalving(xgb, params, cv=5,
                           ratio=2, force_exhaust_budget=True, scoring='roc_auc', verbose=3, n_jobs=6
                           ).fit(data_X, data_Y)
t1 = time.time()
print(t1 - t0)
results = pd.DataFrame.from_dict(sh.cv_results_)
results.groupby('iter').r_i.unique()
results[['r_i', 'param_max_depth', 'param_n_estimators', 'mean_test_score']]

# random_search = GridSearchCV(xgb, param_grid=params,  scoring='roc_auc', n_jobs=6, cv=5, verbose=3)
# random_search.fit(data_X, data_Y)

# # Correls
# data_f = data_2.select_dtypes(include=['float64'])
#
# plt.figure(figsize=(10, 10))
# plt.matshow(data_f.corr(), fignum=1)
# plt.show()
