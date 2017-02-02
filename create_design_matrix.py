# requires: html5lib, lxml
import logging
import glob
import seaborn as sns # for easy on the eyes defaults
import pandas as pd
import re
from zipfile import ZipFile
from io import BytesIO
import requests
import matplotlib.pyplot as plt
import os
import math
import numpy as np
from distutils.version import LooseVersion

import ipdb

assert LooseVersion( pd.__version__) >= LooseVersion('0.19'), "Use Pandas version >= 0.19 on Mac, data parsing depends on order and a few defaults in Pandas library for correctness."

# ONET Version specific constants, we
# download those version between LAST/First_ONET_DB
# and change the download url formatting for older versions
db_releases_url = "http://www.onetcenter.org/db_releases.html" # d/l link isn't working!?
OLD_ONET_VERSION = 20.0
LAST_ONET_DB = 21.1
FIRST_ONET_DB = 14.0
ONET_3_0_DB_DATE = "2000-08-01"

named_tables=["Task Ratings.txt", "Task Statements.txt"]
file_name_task_dwa = "Tasks to DWAs.txt"
output_dir = "./db_releases/"
output_matrix_name = 'design_matrix.csv'

def extract_version_date(version_string):
    ret = re.search("\d+\.\d+", version_string)

    if not ret: # August 2000 ONET 98 version most likely, note this breaks ordering
        ret = re.search("\d+", version_string)

    return ret.group()

def construct_dl_url(row,
                     version,
                     old_onet_version = OLD_ONET_VERSION,
                     last_onet_db = LAST_ONET_DB,
                     first_onet_db = FIRST_ONET_DB,
                     prefix="http://www.onetcenter.org/dl_files/",
                     directory='database/',
                     file_prefix='db_',
                     postfix="_text",
                     extension=".zip"):

    ret = None

    if float(version) <= old_onet_version:
        directory = ""
        postfix = ""

    version = version.split('.') # for use in url concatonation

    dl_url = ''.join((prefix,
                      directory,
                      file_prefix,
                      "{}_{}".format(version[0], version[1]),
                      postfix,
                      extension))

    ret = {'version': float('.'.join(version)),
           'url': dl_url}

    return ret

def extract_named_tables(dl_url, named_tables=["Task Ratings.txt", "Task Statements.txt"]):
    ret = {'dl_url': dl_url,
           'dfs': {} }

    logging.info("\tAbout to fetch {}".format(dl_url))
    tables = requests.get(dl_url)
    with ZipFile(BytesIO(tables.content)) as obj:
        for name in named_tables:
            logging.info("\t\tAbout to extract {}".format(name))
            path = obj.namelist()[0] # assume one directory deep, first one
            ret['dfs'][name] = pd.read_csv( obj.open("".join((path, name))), sep='\t')
            logging.info("\t\tExtracted {}".format(name))

    logging.info("\tFetched {}".format(dl_url))

    return ret

# Pull down ONET related date iff `db_releases` does not exist
if not os.path.isdir(output_dir):
    versioned_releases = pd.io.html.read_html(db_releases_url)[0][0]

    # Extract, Transform Stages for ONET databaes releases and named tables extracted
    os.makedirs(output_dir, exist_ok=True)

    for idx, row in versioned_releases.iteritems():
        version = extract_version_date(row)
        if float(version) <= LAST_ONET_DB and float(version) >= FIRST_ONET_DB:
            dl_url = construct_dl_url(row, version)
            print(dl_url['url'])
            tables = extract_named_tables(dl_url['url'], named_tables=named_tables)

            for table_name in tables['dfs']:
                df = tables['dfs'][table_name]
                file_name = "".join((output_dir, '_'.join((version, table_name))))
                df.to_csv(file_name, sep='\t')

    # ... merge together tasks across database releases
    selected_idx = 0
    selected_filenames = glob.glob(output_dir + '*' + named_tables[selected_idx] + '*')
    merged_table = pd.concat((pd.read_csv(f, sep='\t') for f in selected_filenames),
                              ignore_index=True)

    # todo: drop indices from being written
    merged_table['Date'] = pd.to_datetime(merged_table['Date'], format="%m/%Y")
    merged_table.to_csv('merged_'+named_tables[selected_idx], sep='\t')

    # ... store off latest Task to DWA table as well (note: this table was added as a new file
    # for release 18.1, and kept the same for release 19.0 onward. Here we assume that
    # the current release will have all current and older (if any) task ides.
    #
    # we also assume the IWAs can be extracted from the DWA ID field as well (preventing us from needing
    # to d/l yet another table)
    dl_url = construct_dl_url(versioned_releases[0], version, extension="/"+file_name_task_dwa)
    tasks_to_dwas = pd.read_csv( BytesIO( requests.get(dl_url['url']).content), sep="\t")

    tasks_to_dwas["IWA ID"] = tasks_to_dwas['DWA ID'].str[:-4] #fixed width DWA ID format
    tasks_to_dwas.to_csv(file_name_task_dwa, sep="\t")

# Analysis Stage

# read in data tables on which to do analysis on...
tasks_to_dwas = pd.read_csv( file_name_task_dwa, sep="\t")

selected_idx = 0 # we'll just take the first merged table, Task Ratings.txt
merged_table = pd.read_csv('merged_'+named_tables[selected_idx], sep='\t')
merged_table['Date'] = pd.to_datetime(merged_table['Date'])

# Question 1: How temporally distinct are the task updates where more than 1 update happend?
#
# Why: To identify task frequency decline/ascent we need at least two samples in time.
# Here the operating assumptions is that between those time before/after samples:
#     a) The US workforce was affected by AI/Automation
#     b) Some AI/Automation can affect human work activities through replacement
#     c) Tasks affected by AI/Automation replacement will be carried out less frequently, as
#        evidenced by temporal decline in frequency across occupations. There will be something similar
#        about affected tasks/IWAs over those not affected.
#
#     To point to the casual effect of AI/Automation between the time samples as being generally plausible
#     we should verify that each set of time samples are generally clustered together. If the samples are
#     difused across time then we can't point to the intervening effect of AI/Automation because the
#     before and after samples occur during the intervening effect.

# Although we work at the IWA level for the higher level analysis (not yet described), we work at the Task
# level here because it's simpler and the fundamental component of something AI/Automation replaces.

# ... So, given above, we first want to get those tasks that have occurred at least 2 times

RUN_Q1 = False

if RUN_Q1:
    unique_tasks_by_date = merged_table.drop_duplicates(subset=['Task ID', 'Date'], keep='first')
    task_updates_freq = unique_tasks_by_date['Task ID'].value_counts()
#update_mask = task_updates_freq > 1

# ... this duplicates frequency data but makes it much easier to analyze and manipulate things
# todo: clear SettingWithCopy warning (see is_copy, set to false?)
#? unique_tasks_by_date.is_copy = False
    unique_tasks_by_date['frequency'] = unique_tasks_by_date['Task ID'].map(task_updates_freq)

# ... now we want to plot these tasks across time with the hope of observing a gap of time betwen two
# clusters of before/after samples. So this is an overlapping density plot of the two samples.
    mask = unique_tasks_by_date['frequency'] > 1

# ... to capture sample points in time we take teh cumulative count across Task ID, so that
# cumulative count = 1 is the first sample of that Id, c.count = 2 is the second and so on.
# there are at most 3 samples taking, with most having only 2

# note: get rid of annoying value is trying to be set on copy warning
    unique_tasks_by_date['cumulative counts'] = unique_tasks_by_date.groupby("Task ID").cumcount()+1

# ... now we plot overlapping histograms of the before, aftre sampling in time. We should see
# distinct clusters
    before_mask = unique_tasks_by_date['cumulative counts'] == 1
    after_mask = unique_tasks_by_date['cumulative counts'] == 2
    beyond_mask = unique_tasks_by_date['cumulative counts'] == 3

# The analysis stage is sensitive to the OS used; this was done on MacOS
    bins = 50
    alpha = 0.5
    plt.hist(unique_tasks_by_date.loc[after_mask, 'Date'].values, color='orange', bins=bins, width=bins*1, alpha=2*alpha, label='after')
    plt.hist(unique_tasks_by_date.loc[before_mask, 'Date'].values, color='blue', bins=bins, alpha=alpha, label='before')
    plt.hist(unique_tasks_by_date.loc[beyond_mask, 'Date'].values, color='pink', bins=bins, width=bins*1.25, alpha=alpha, label='beyond')
    plt.legend(loc='upper right')
    plt.ion()
    plt.show()

# ... yep, we see a seperation of before and after task samples at about 2009 - 2010.
# so we'll use this point in time to index all Task IDs with samples before and after that point in time

# 2) Construct the Design Matrix, following Issue 1

# Reformat the merged table to contain only 'FT' related rows, deduplicated by Task ID/Date,
# drop/replaced unneeded values
mt = merged_table.drop(merged_table.columns[[0,1,11]], 1)
mt = mt.convert_objects(convert_numeric=True) # depreciated
mt = mt.drop_duplicates(subset=['Task ID', 'Date', 'Category'], keep='first')
mt = mt[mt['Scale ID'] == 'FT']
mt = mt.drop('Scale ID', 1) # don't need this any more
mt = mt.replace('n/a', np.NaN)

# Add in computed year hours, sort dataframe to answer on change in hours, days per task across years
# make everything numbery like
mt = mt.sort_values(by=['Task ID', 'Date'])

# convert each task category grouping into total hours per year, by this formula
#
# Time1hours =(E2 * 0.01 * 0.5 * 0.45) + (E4 * 0.01 * 1 * 0.45) + (E6 * 0.01 * 12 * 0.45) + (E8 * 0.01 * 52 * 0.45) + (E10 * 0.01 * 260 *0.45)  + (E12 * 0.01 * 520 *0.45)  + (E14 * 0.01 * 1043 * 0.45)
# E* refers to order of category values
adjusted_scale = 0.45
percent_scale = 0.01
category_factors = [0.5, 1, 12, 52, 260, 520, 1043]

# group
grouped = mt.groupby(['Task ID', 'Date'])

# aggregate by Task ID/Date by calculating hours per year
def hours_per_year(hours, adjusted_scale=adjusted_scale, category_factors=category_factors):
    return (hours*category_factors).sum() * adjusted_scale * percent_scale

agg = grouped.agg({'Data Value': hours_per_year})

# fix this code, kinda wonky, special cased but apparently gets the job done
# "unroll" groups into one line, include other Task ID row information
agg.reset_index(inplace=True)

# What we need to select first indices and merge on the reset
first = agg.drop_duplicates('Task ID', keep='first').index
others = agg.index.difference(first)

agg = agg.loc[first].merge(agg.loc[others], on="Task ID", suffixes=('1', '2'), copy=False)

# join with IWA Ids
agg = agg.merge(tasks_to_dwas[["Task ID", "IWA ID"]], on="Task ID")
agg = agg[['Task ID', 'Date1', 'Data Value1', 'Date2', 'Data Value2', 'IWA ID']] # todo: make so this isn't required

# todo: make as transform call?
agg['difference in hours'] = agg['Data Value2']-agg['Data Value1']
agg['difference in days'] = agg['Date2']-agg['Date1']
agg['difference in days'] = (agg['difference in days']/np.timedelta64(1, 'D')).astype(int)

#  output to disk
agg.to_csv('./design_matrix/design_matrix.csv', sep='\t')

# (add to agg)
# Tables with additional features, supplied outside of repository
tmdata = pd.read_csv("../task_model_data.bsv", sep="|")
iwa_bing = pd.read_csv('../31jandata/iwa_model_data.bsv', sep="|") # cd to ../31jandata

agg =\
     agg.merge(iwa_bing[["IWA_ID", "IWA_DESC", "bing_hits", "social_index", "creative_index", "pm_index", "log_bing"]],
                              right_on="IWA_ID",
                              left_on="IWA ID")
agg =\
     agg.merge(tmdata[['IWA_ID', 'relevance', 'importance', 'job_zone']], right_on="IWA_ID", left_on="IWA ID")

cols_to_keep =\
      ['Task ID', 'Date1', 'Data Value1', 'Date2', 'Data Value2', 'IWA ID',\
       'difference in hours', 'difference in days',\
       'social_index', 'creative_index', 'pm_index',\
       'relevance', 'importance', 'job_zone',
       'log_bing']
agg =\
     agg[cols_to_keep].dropna() # note that we're droping rows with NAs


# 3) Now we conduct exploratory analysis on random subset
exploratory_agg = agg.sample(frac=0.90, random_state=42)
holdout_agg = agg.drop(exploratory_agg.index)

cols_to_scale = [exploratory_agg.columns[idx] for idx in [2, 4, 6, 7,
                                                          8, 9, 10, 11,
                                                          12, 13, 14]]
from sklearn.preprocessing import RobustScaler
robust_scaler = RobustScaler()

# not needed
#exploratory_agg[cols_to_scale] =\
#        robust_scaler.fit_transform(exploratory_agg[cols_to_scale])

# plot results
PLOT_JOINT = True
if PLOT_JOINT: # takes a while
    g = sns.PairGrid(exploratory_agg[cols_to_scale])
    g.map_diag(sns.kdeplot)
    g.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=6)

g = ['Data Value1', 'social_index', 'pm_index', 'relevance', 'importance', 'job_zone', 'log_bing']
from sklearn.tree import ExtraTreesRegressor

et = ExtraTreesRegressor(n_estimators=10, random_state=0, n_jobs=2)

et = et.fit(exploratory_agg[g], exploratory_agg['difference in hours'])
et.score(holdout_agg[g], holdout_agg['difference in hours'])
# R^2 0.93 on unseen data

# need treeinterpreter to conditionally interpret variable importances, similarly to coef in a linear model
#   also need interaction importances, which can be interpreted by the same package
