#!/usr/bin/env python

import pandas as pd
import numpy as np
import time
import datetime
from sets import Set
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression



def bag_of_words(X):
	# list of distinct heroes sorted by id
	heroes = set()
	for i in range(5):
		heroes |=  set(X['r%d_hero' % (i+1)].value_counts().index)
		heroes |=  set(X['d%d_hero' % (i+1)].value_counts().index)
	heroes = sorted(list(heroes))
	N_hero = len(heroes)
	#print '#heroes=%d' % N_hero

	# array of hero's participation in match
	# part[m_id, h_id] = p
	# p = 0 if hero h_id is not participating in match m_id
	# p = 1 if hero h_id is playing for Radiant during match m_id
	# p = -1 if hero h_id is playing for Dire during match m_id
	part = np.zeros((X.shape[0], N_hero))

	for m_id, match_id in enumerate(X.index.values):
		for p in range(5):
			hero = X.ix[match_id, 'r%d_hero' % (p+1)]
			h_id = heroes.index(hero)
			part[m_id, h_id] = 1
		
			hero = X.ix[match_id, 'd%d_hero' % (p+1)]
			h_id = heroes.index(hero)
			part[m_id, h_id] = -1

	# turn array into DataFrame and add it to the design matrix
	X_part = pd.DataFrame(part, index=X.index.values, columns=heroes)
	X = pd.concat([X, X_part], axis=1)
	return X



if __name__ == '__main__':
	# import data
	data = pd.read_csv('./features.csv', index_col='match_id')

	# target vector
	y = data['radiant_win']

	# NA->0 althought there's probably a better choice
	data = data.fillna(0)

	# seed
	seed = 123

	X = data
	
	# include categorial data appropriately
	X = bag_of_words(X)
	
	# remove features with lookahead bias
	drop_col = [	'duration',
					'radiant_win',
					'tower_status_radiant',
					'tower_status_dire',
					'barracks_status_radiant',
					'barracks_status_dire',
					'lobby_type']
	for i in range(5):
		drop_col.append('r%d_hero' % (i+1))
		drop_col.append('d%d_hero' % (i+1))
	X.drop(drop_col, 1, inplace=True)
	
	# stats[logC] = cv_score
	stats = dict()
	
	kf = KFold(n_splits=5, random_state=seed, shuffle=True)
	
	for logC in range(-6, 1):
		C = pow(10, logC)
		
		start_time = datetime.datetime.now()
		
		clf = make_pipeline(StandardScaler(), LogisticRegression(C=C))
		
		cv = cross_val_score(estimator=clf, X=X, y=y, scoring='roc_auc', cv=kf)
		
		print '\nlog10(C) = %d\nCross-validation score (mean) = %0.5f\nTime (sec): %0.1f' % (logC, cv.mean(), (datetime.datetime.now() - start_time).total_seconds())
		
		stats[logC] = cv.mean()
	
	key_max = max(stats, key=stats.get)

	logc, cv_max = key_max, stats[key_max]
	
	print '\nMax value of cross-validation score %0.5f is attained on log10(C) = %d' % (cv_max, logc)
