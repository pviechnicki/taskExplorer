# requires: html5lib, lxml
import logging
import glob
import seaborn as sns # for easy on the eyes defaults
import pandas as pd
import ipdb
import numpy as np
from treeinterpreter import treeinterpreter as ti
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import ExtraTreesRegressor

random_seed = 0
filename = "task_model_data.feb5.bsv"
predicator_variables = ["importance", "relevance", "task_person_hours1",
                        "year_span", "pm_job", "social_job",
                        "social_job", "normalized_job_zone", "pmjob_x_jobzone",
                        "social_x_jobzone"]
response_variable = ["difference_in_hours"]

design_matrix = pd.read_csv(filename, sep="|")[predicator_variables + response_variable]

et = ExtraTreesRegressor(n_estimators=10, random_state=0, n_jobs=2)
regressor = RandomForestRegressor(n_estimators=10,
                                  random_state=random_seed,
                                  n_jobs=2)

clf = regressor #et
cv = ShuffleSplit(n_splits=3,
                  test_size=0.05,
                  random_state=random_seed)

# double check that I'm using this correctly...
scores = cross_val_score(clf,
                         design_matrix[predicator_variables],
                         np.ravel(design_matrix[response_variable].values),
                         cv=cv,
                         n_jobs=2)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#
#cv = ShuffleSplit(n_splits=3,
#                  test_size=0.05,
#                  random_state=random_seed)
#
#scores = []
#for train, test in cv(design_matrix):
#    clf.fit(design.iloc[train,])
#    clt.predict(test
