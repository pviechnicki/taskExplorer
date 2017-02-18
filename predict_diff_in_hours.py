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

from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score

# quick train/test script for testing on various task data sets against various algorithms

random_seed = 0
test_size = 0.10

feb5 = {'filename':"task_model_data.feb5.bsv",
        'predicator_variables':["importance", "relevance", "task_person_hours1",
                                "year_span", "pm_job", "social_job",
                                "social_job", "normalized_job_zone", "pmjob_x_jobzone",
                                "social_x_jobzone"],
        'response_variable' : ["difference_in_hours"],
        'sep'="|"}

feb18 = {'filename':"task_model_traiing_data.feb18.bsv",
         'predicator_variables':["importance", "relevance", "task_person_hours1",
                                "year_span", "pm_job", "social_job",
                                "social_job", "normalized_job_zone", "pmjob_x_jobzone",
                                "social_x_jobzone"],
         'response_variable' : ["difference_in_hours"],
         'sep'='|'}

orig = {'filename':"./design_matrix/design_matrix.csv",
        'predicator_variables':["importance", "relevance", "task_person_hours1",
                                "year_span", "pm_job", "social_job",
                                "social_job", "normalized_job_zone", "pmjob_x_jobzone",
                                "social_x_jobzone"],
        'response_variable' : ["difference in hours"],
        'sep'='\t'}

dataset_info = [feb5, feb18, orig]

for info in dataset_info:
    filename = info['filename']
    predicator_variables = info['predicator_variables']
    response_variable = info['response_variable']
    sep = info['sep']

    design_matrix = pd.read_csv(filename, sep=sep)[predicator_variables + response_variable]

    et = ExtraTreesRegressor(n_estimators=10, random_state=0, n_jobs=2)
    regressor = RandomForestRegressor(n_estimators=10,
                                      random_state=random_seed,
                                      n_jobs=2)

    clf = regressor #et
    cv = ShuffleSplit(n_splits=3,
                      test_size=test_size,
                      random_state=random_seed)

# double check that I'm using this correctly...
    scores = cross_val_score(clf,
                             design_matrix[predicator_variables],
                             np.ravel(design_matrix[response_variable].values),
                             cv=cv,
                             n_jobs=2)

    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    cv = ShuffleSplit(n_splits=3,
                      test_size=test_size,
                      random_state=random_seed)

# same as accuracy
    scores = cross_val_score(clf,
                             design_matrix[predicator_variables],
                             np.ravel(design_matrix[response_variable].values),
                             cv=cv,
                             n_jobs=2,
                             scoring=make_scorer(r2_score))

    print("R^2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
