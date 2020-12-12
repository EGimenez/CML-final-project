import pandas as pd
from src.data_management import my_imputer, prep_data, stay_num, drop, mark_missing, dummify, my_pca
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from dabl.search import GridSuccessiveHalving
import time
import pickle
from src.XGBooster import get_xgbooster
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Params
CALIBRATE_DEATH = False
SHORT_LONG = True
ShortModel_0_Admission_Elective_0 = True
ShortModel_0_Admission_Elective_1 = True
ShortModel_1_Admission_Elective_0 = True
ShortModel_1_Admission_Elective_1 = True

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

# BEGIN
data_2 = data_join_2[:len(data)]
index_0 = data_2.index

data_2 = data_2.sample(frac=0.8, axis=0)
index_s = data_2.index
index_r = set(index_0) - set(index_s)
kaggle_2 = data_join_2[:len(data)].loc[index_r, :]
# END

# Forward_columns
columns_drop = ['hadm_id', 'icustay_id', 'subject_id',
                        'SPECIAL_ADMIN',
                        'MY_ADMIN_DATE',
                        'MY_CHECKOUT_DATE', 'MY_CHECKOUT_TIME', 'MY_CHECKOUT_WEEKDAY', 'MY_CHECKOUT_WEEKDAY_DEC',
                        'SPECIAL_ADMIN',
                        'MY_LOS', 'LOS_DEC', 'LOS']

estimated_columns = ['SHORT_LONG_MODEL', 'HOSPITAL_EXPIRE_FLAG']


# Caracterizamos Death
data_X = drop(data_2, columns_drop + estimated_columns, is_reg=True)
data_Y = data_2['HOSPITAL_EXPIRE_FLAG']
params = {'max_depth': [5, 6], 'n_estimators': range(400, 700, 50)}
params = {'min_child_weight': [1, 5, 10],
          'gamma': [0.5, 1, 1.5, 2, 5],
          'max_depth': [5, 6, 7],
          'n_estimators': range(300, 750, 50)}
sh_d = get_xgbooster(data_X, data_Y, 'death', params, CALIBRATE_DEATH)
kaggle_2['HOSPITAL_EXPIRE_FLAG'] = sh_d.predict(drop(kaggle_2, columns_drop + estimated_columns, is_reg=True))

# Caracterizamos ShortLong
data_X = drop(data_2, columns_drop + ['SHORT_LONG_MODEL'], is_reg=True)
data_Y = data_2['SHORT_LONG_MODEL']
params = {'max_depth': [5, 6], 'n_estimators': range(400, 700, 50)}
params = {'gamma': [0.5, 1, 1.5, 2, 5],
          'max_depth': [5, 6, 7],
          'n_estimators': range(300, 700, 50)}
sh_s = get_xgbooster(data_X, data_Y, 'short_long', params, SHORT_LONG)
kaggle_2['SHORT_LONG_MODEL'] = sh_s.predict(drop(kaggle_2, columns_drop + ['SHORT_LONG_MODEL'], is_reg=True))


# Hey Ho Let's Go
result = pd.DataFrame(kaggle_2['icustay_id'])
Y_train_MY_LOS = data_2['MY_LOS']
Y_train_LOS = data_2['LOS']
Y_train_prob = data_2['SHORT_LONG_MODEL']

X_train = drop(data_2, columns_drop, is_reg=True)
X_test  = drop(kaggle_2, columns_drop, is_reg=True)

# ShortModel_0_Admission_Elective_0
params = {'max_depth': [5, 6], 'n_estimators': range(400, 700, 50)}
params = {'min_child_weight': [1, 5, 10],
          'gamma': [0.5, 1, 1.5, 2, 5],
          'max_depth': [5, 6, 7],
          'n_estimators': range(300, 750, 50)}
index = X_train[(X_train['SHORT_LONG_MODEL'] == 0) & (X_train['ADMISSION_TYPE_ELECTIVE'] == 0)].index
index = X_train[(X_train['ADMISSION_TYPE_ELECTIVE'] == 0)].index
sh_00 = get_xgbooster(X_train.loc[index, :], Y_train_LOS[index], 'ShortModel_0_Admission_Elective_0', params, ShortModel_0_Admission_Elective_0)

# ShortModel_0_Admission_Elective_1
params = {'max_depth': [5, 6], 'n_estimators': range(400, 700, 50)}
params = {'min_child_weight': [1, 5, 10],
          'gamma': [0.5, 1, 1.5, 2, 5],
          'max_depth': [5, 6, 7],
          'n_estimators': range(300, 750, 50)}
index = X_train[(X_train['SHORT_LONG_MODEL'] == 0) & (X_train['ADMISSION_TYPE_ELECTIVE'] == 1)].index
index = X_train[(X_train['ADMISSION_TYPE_ELECTIVE'] == 1)].index
sh_01 = get_xgbooster(X_train.loc[index, :], Y_train_LOS[index], 'ShortModel_0_Admission_Elective_1', params, ShortModel_0_Admission_Elective_1)

# ShortModel_1_Admission_Elective_0
params = {'max_depth': [5, 6], 'n_estimators': range(400, 700, 50)}
params = {'min_child_weight': [1, 5, 10],
          'gamma': [0.5, 1, 1.5, 2, 5],
          'max_depth': [5, 6, 7],
          'n_estimators': range(300, 750, 50)}
index = X_train[(X_train['SHORT_LONG_MODEL'] == 1) & (X_train['ADMISSION_TYPE_ELECTIVE'] == 0)].index
index = X_train[(X_train['ADMISSION_TYPE_ELECTIVE'] == 0)].index
sh_10 = get_xgbooster(X_train.loc[index, :], Y_train_MY_LOS[index], 'ShortModel_1_Admission_Elective_0', params, ShortModel_1_Admission_Elective_0)

# ShortModel_1_Admission_Elective_1
params = {'max_depth': [5, 6], 'n_estimators': range(400, 700, 50)}
params = {'min_child_weight': [1, 5, 10],
          'gamma': [0.5, 1, 1.5, 2, 5],
          'max_depth': [5, 6, 7],
          'n_estimators': range(300, 750, 50)}
index = X_train[(X_train['SHORT_LONG_MODEL'] == 1) & (X_train['ADMISSION_TYPE_ELECTIVE'] == 1)].index
index = X_train[(X_train['ADMISSION_TYPE_ELECTIVE'] == 1)].index
sh_11 = get_xgbooster(X_train.loc[index, :], Y_train_MY_LOS[index], 'ShortModel_1_Admission_Elective_1', params, ShortModel_1_Admission_Elective_1)

result['LOS'] = 0

# ShortModel_0_Admission_Elective_0
index = X_test[(X_test['SHORT_LONG_MODEL'] == 0) & (X_test['ADMISSION_TYPE_ELECTIVE'] == 0)].index
result.loc[index, ['LOS']] = sh_00.predict(X_test.loc[index, :]) - kaggle_2.loc[index, ['MY_ADMIN_TIME']]['MY_ADMIN_TIME']/24

# ShortModel_0_Admission_Elective_1
index = X_test[(X_test['SHORT_LONG_MODEL'] == 0) & (X_test['ADMISSION_TYPE_ELECTIVE'] == 1)].index
result.loc[index, ['LOS']] = sh_01.predict(X_test.loc[index, :]) - kaggle_2.loc[index, ['MY_ADMIN_TIME']]['MY_ADMIN_TIME']/24

# ShortModel_1_Admission_Elective_0
index = X_test[(X_test['SHORT_LONG_MODEL'] == 1) & (X_test['ADMISSION_TYPE_ELECTIVE'] == 0)].index
result.loc[index, ['LOS']] = sh_10.predict(X_test.loc[index, :])

# ShortModel_1_Admission_Elective_1
index = X_test[(X_test['SHORT_LONG_MODEL'] == 1) & (X_test['ADMISSION_TYPE_ELECTIVE'] == 1)].index
result.loc[index, ['LOS']] = sh_11.predict(X_test.loc[index, :])

result.to_csv('result.csv', index=False)



# Other
index = X_test[(X_test['ADMISSION_TYPE_ELECTIVE'] == 0)].index
prob = sh_s.predict_proba(drop(X_test.loc[index, :], estimated_columns, is_reg=True))
result.loc[index, ['LOS']] = prob[:, 0]*(sh_00.predict(X_test.loc[index, :]) - kaggle_2.loc[index, ['MY_ADMIN_TIME']]['MY_ADMIN_TIME']/24) + \
                             prob[:, 1]*sh_10.predict(X_test.loc[index, :])


index = X_test[(X_test['ADMISSION_TYPE_ELECTIVE'] == 1)].index
prob = sh_s.predict_proba(drop(X_test.loc[index, :], estimated_columns, is_reg=True))
result.loc[index, ['LOS']] = prob[:, 0]*(sh_01.predict(X_test.loc[index, :]) - kaggle_2.loc[index, ['MY_ADMIN_TIME']]['MY_ADMIN_TIME']/24) + \
                             prob[:, 1]*sh_11.predict(X_test.loc[index, :])

result.to_csv('result_3.csv', index=False)

res = mean_squared_error(data_join_2[:len(data)].loc[index_r, ['LOS']]['LOS'], result['LOS'])

plt.plot(data_join_2[:len(data)].loc[index_r, ['LOS']]['LOS'], result['LOS'], 'o',  markersize=0.5)
plt.show()
print(res**0.5)
print('Hola')

