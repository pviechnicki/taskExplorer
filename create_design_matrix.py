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
mt = mt.drop_duplicates(subset=['Task ID', 'Date', 'Category'], keep='first')
mt = mt[mt['Scale ID'] == 'FT']
mt = mt.drop('Scale ID', 1) # don't need this any more
mt = mt.replace('n/a', np.NaN)

# Add in computed year hours, sort dataframe to answer on change in hours, days per task across years
# make everything numbery like
hours_map = {'1':0.5, '2':1, '3':12, '4':52, '5':260, '6':520, '7':1043}
scale_factor = 0.45
mt['hours per year'] = mt['Category'].map(hours_map)*scale_factor
mt = mt.sort_values(by=['Task ID', 'Category', 'Date'])
mt = mt.convert_objects(convert_numeric=True) # depreciated

# For Task ID, Category chunks take the diff of several columns:
#   hours per year, Standard Error, Date (diff_colums), diffed into diff_variables
diff_columns = ['days offset','change hours per year','change in SE']
diff_variables = ['Date', 'hours per year', 'Standard Error']

mt[diff_columns] = mt.groupby(['Task ID', 'Category'])[diff_variables].diff()

# output interim result
mt.to_csv('./design_matrix/design_matrix.csv', sep='\t')

# TODO attach IWA numbers, break out raw per IWA

#tasks_to_dwas[]
# figure out what kind of join will make a row for each unique key in the right table
