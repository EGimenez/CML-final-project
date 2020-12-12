import pandas as pd
from src.data_management import my_imputer, prep_data, stay_num, drop, mark_missing, dummify, my_pca
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from dabl.search import GridSuccessiveHalving
import time
import pickle
import xgboost

def get_xgbooster(X, Y, name, params, calculate):
	if calculate:
		if len(Y.unique()) == 2:
			xgb = xgboost.XGBClassifier(learning_rate=0.02, param_max_depth=5, n_estimators=400, scoring='roc_auc', objective='binary:logistic',
		                    silent=True, nthread=1)
			t0 = time.time()
			sh = GridSuccessiveHalving(xgb, params, cv=5, ratio=2, force_exhaust_budget=True, scoring='roc_auc', verbose=3, n_jobs=6).fit(X, Y)
			t1 = time.time()
		else:
			xgb = xgboost.XGBRegressor(learning_rate=0.02, param_max_depth=5, n_estimators=400, objective ='reg:linear',
		                    silent=True, nthread=1)
			t0 = time.time()
			sh = GridSuccessiveHalving(xgb, params, cv=5, ratio=2, force_exhaust_budget=True, scoring='neg_mean_squared_error',  verbose=3, n_jobs=6).fit(X, Y)
			t1 = time.time()

		# Halving

		print(t1 - t0)

		results = pd.DataFrame.from_dict(sh.cv_results_)
		results.groupby('iter').r_i.unique()
		results[['r_i'] + ['param_' + k for k in params.keys()] + ['mean_test_score']].to_csv(name+'.csv')

		with open(name+'.pickle', 'wb') as input:
			pickle.dump(sh, input, protocol=pickle.HIGHEST_PROTOCOL)
	else:
		with open(name+'.pickle', 'rb') as output:
			sh = pickle.load(output)
	return sh