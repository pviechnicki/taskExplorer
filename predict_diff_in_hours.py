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
from sklearn import linear_model # seems to hang and not work?
from sklearn.linear_model.stochastic_gradient import SGDRegressor
import warnings

from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score

#note: user should have most up to date numpy, scipy and sklearn
# to avoid BLAS/LaPack errors when using linear_model.LinearRegression()

# quick train/test script for testing on various task data sets against various algorithms
# broken up into Functions, Variables and a driver loop (for loop over classifier, data sets)

# Run initalization
# ignore warning, see: https://github.com/scipy/scipy/issues/5998
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

# Functions
def make_cv(test_size, random_seed, n_splits=2):
    cv = ShuffleSplit(n_splits=n_splits,
                      test_size=test_size,
                      random_state=random_seed)
    return cv

def cross_validate_me(clf,
                      test_size,
                      design_matrix_name,
                      design_matrix,
                      random_seed,
                      predicator_variables,
                      response_variable):
    print("\n\t-----------------------------")
    print("\t File: {}".format(design_matrix_name))
    print("\t Learner Type: ", clf)
    print("\t-----------------------------")

    scores = cross_val_score(clf,
                             design_matrix[predicator_variables],
                             np.ravel(design_matrix[response_variable].values),
                             cv=make_cv(test_size, random_seed),
                             n_jobs=2)
    print("%d-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (len(scores), scores.mean(), scores.std() * 2))

    # same as accuracy
    #scores = cross_val_score(clf,
    #                         design_matrix[predicator_variables],
    #                         np.ravel(design_matrix[response_variable].values),
    #                         cv=make_cv(test_size, random_seed),
    #                         n_jobs=2,
    #                         scoring=make_scorer(r2_score))

    #print("(%d-fold average) R^2: %0.2f (+/- %0.2f)" % (len(scores), scores.mean(), scores.std() * 2))

    #print("Predictor Variables Used: (response variable {})".format(response_variable[0]),
    #      "\n\t",
    #      "\n\t".join(predicator_variables))
    print("\t-----------------------------")

# Variables
random_seed = 0
test_size = 0.10
year_span_range = 10 # predict on year_span = 1..year_span_range

orig = {'filename':"./design_matrix/design_matrix_task_model_bing.csv",
        'predicator_variables':['Data Value1', 'social_index', 'pm_index',
                                'relevance', 'importance', 'job_zone',
                                'log_bing'],
        'response_variable' : ["difference in hours"],
        'sep':'\t'}

orig = {'filename':"./design_matrix/design_matrix_task_model_bing.csv",
        'predicator_variables':['Data Value1', 'social_index', 'pm_index',
                                'relevance', 'importance', 'job_zone',
                                'log_bing'],
        'response_variable' : ["difference in hours"],
        'sep':'\t'}

feb5 = {'filename':"task_model_data.feb5.bsv",
        'predicator_variables':["importance", "relevance", "task_person_hours1",
                                "year_span", "pm_job", "social_job",
                                "social_job", "normalized_job_zone", "pmjob_x_jobzone",
                                "social_x_jobzone"],
        'response_variable' : ["difference_in_hours"],
        'sep':"|"}

feb18 = {'filename':"task_model_training_data.feb18.bsv",
         'predicator_variables':["importance", "relevance", "normalized_job_zone",
                                 "year_span", "social_job", "creative_job",
                                 "pm_job", "automation_index"],
         'response_variable' : ["difference_in_hours"],
         'sep':'|'}

predictme = {'filename':"task_forecast_full_data.bsv",
             'predicator_variables':["importance", "relevance", "normalized_job_zone",
                                     "year_span", "social_job", "creative_job",
                                     "social_job", "creative_job",
                                     "pm_job", "automation_index"],
             'response_variable' : ["difference_in_hours"],
             'sep':'|'}

#dataset_info = [feb18, feb5, feb18, orig]
dataset_info = [orig]
dataset_predict = [] # [predictme]

# Driver loop (runs data, classifers over another, measuring accuracy, etc)
for info in dataset_info:  # cross validate set of regressors over each dataset
    filename = info['filename']
    predicator_variables = info['predicator_variables']
    response_variable = info['response_variable']
    sep = info['sep']

    design_matrix = pd.read_csv(filename, sep=sep)[predicator_variables + response_variable]
    design_matrix.drop_duplicates(inplace=True)
    print("\t(dropped duplicates)")

    regressors = [\
        RandomForestRegressor(n_estimators=10,
                              random_state=random_seed,
                              criterion="mse",
                              max_depth=None,
                              oob_score=True,
                              n_jobs=4)]

    for regressor in regressors:
        cross_validate_me(regressor,
                          test_size = test_size,
                          design_matrix = design_matrix,
                          design_matrix_name = filename,
                          random_seed = random_seed,
                          predicator_variables = predicator_variables,
                          response_variable = response_variable)

        # Output predictions, accuracy should be somethign like what we see in
        # cross validate
        print("\n training on entire ", filename)
        trained = regressor.fit(design_matrix[predicator_variables],
                                np.ravel(design_matrix[response_variable].values))
        for predict in dataset_predict:
            filename = predict['filename']
            predicator_variables = predict['predicator_variables']
            response_variable = predict['response_variable']
            sep = predict['sep']

            print("\n predicting on ", filename)
            predict_matrix = pd.read_csv(filename, sep=sep)[predicator_variables]
            predict_matrix[response_variable[0]] = trained.predict(predict_matrix[predicator_variables])
            predict_matrix.to_csv(filename+".predict", sep=sep)

            for year_span in range(1, year_span_range):
                print(".. with year_span ", year_span)
                predict_matrix['year_span'] = year_span
                predict_matrix[response_variable[0]] = trained.predict(
                                            predict_matrix[predicator_variables + ['year_span']])
                predict_matrix.to_csv(filename+"."+str(year_span) +".predict", sep=sep)
