#!/usr/bin/env python

import pandas as pd
import numpy as np
import time
import datetime
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score


if __name__ == '__main__':
	# import data
	data = pd.read_csv('./features.csv', index_col='match_id')

	# target vector
	y = data['radiant_win']

	# remove features with lookahead bias
	drop_col = [	'duration',
					'radiant_win',
					'tower_status_radiant',
					'tower_status_dire',
					'barracks_status_radiant',
					'barracks_status_dire']
	data.drop(drop_col, 1, inplace=True)

	# NA->0 althought there's probably a better choice
	data = data.fillna(0)

	# seed
	seed = 123

	X = data
	
	# dictionary to keep (#estimators, cross-val score)
	stats = dict()
	
	kf = KFold(n_splits=5, random_state=seed, shuffle=True)
	
	for n_est in [10,20,30]:
		start_time = datetime.datetime.now()
		
		clf = GradientBoostingClassifier(	n_estimators=n_est, 
											random_state=seed,
											subsample=1.0,
											max_depth=3)
		
		cv = cross_val_score(	estimator=clf,
								X=X,
								y=y,
								scoring='roc_auc',
								cv=kf)
		
		print '\n#Estimators = %d\nCross-validation score (mean) = %0.5f\nTime (sec): %0.1f' % (n_est, cv.mean(), (datetime.datetime.now() - start_time).total_seconds())
		
		stats[n_est] = cv.mean()
	
	key_max = max(stats, key=stats.get)
	
	n_est, cv_max = key_max, stats[key_max]
	print '\nMax value of cross-validation score %0.5f is attained on %d estimators' % (cv_max, n_est)
