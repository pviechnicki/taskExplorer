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

# 2) Construct the Design Matrix

# Now we create an index into the merged data table for those task ids that have updates before and after
# the 2009 - 2010 cut off point. This requires that we create masks for those tasks occuring after the date
# cutoff point. The `after_mask` variable is critical in only looking at those tasks that have occured 2x or more
# with a first time before the cutoff date (since it looks at cumlative count == 2 past cut off date).

# This is all done so that we may build up to IWAs (which are composed of multiple Tasks)

# A) Create a set of masks for tasks occuring after the `date_cutoff` so that we can properly select those
# tasks occuring before the cut off and having an update after the cutoff. `resampled_tasks` is a mask for
# those tasks that we want to use as a basis for this analysis.
date_cutoff = '2010-01-01'
after_cutoff = unique_tasks_by_date['Date'] > date_cutoff

# filter out those rows not having a second update (see: after_mask) occuring after 2010-01-01 (see: after_cutoff)

resampled_tasks = merged_table['Task ID'].isin( unique_tasks_by_date.loc[after_mask & after_cutoff, 'Task ID'] )

# (... gives 11,096 tasks (out of 20,200, e.g. merged_table['Task ID'].unique()))
# ... add a column for time differential from task start date

# B) Then we map tasks to the first date that they occured. This helps create the time differential

# ... create a task map *to dates* for those tasks in before_mask that we selected above
before_tasks = unique_tasks_by_date['Date'] <= date_cutoff
# (trick to get a series, appropriate set index, out of a data frame)
task_date_map = unique_tasks_by_date.loc[before_tasks, ['Task ID', 'Date']].set_index('Task ID')['Date']

# C) Following that we start constructing the design matrix which we use for follow on analysis, create
# time related variables (First Task Date, Days Elapsed).

# ... copy off only those rows having tasks that occur before and after cutoff date, add in date/time information ...

task_matrix = merged_table[resampled_tasks]

task_matrix['First Task Date'] = task_matrix['Task ID'].map(task_date_map)
task_matrix['Days Elapsed From First Task Date'] = task_matrix['Date'] - task_matrix['First Task Date']

task_matrix.set_index('Task ID', inplace=True)

# D) As a final step we group task information by IWA, aggregate information for follow on analysis.
# Information aggregated are: date related items (start, differential), Data Rating per frequency. By
# masking before/after the cuttoff date we can crate before and after IWA measurements across time.

# todo: create IWA to Task ID map, reindex task_matrix by Task ID
# For each IWA, for Task IDs in task_matrix masked by time (before or after)
#
# take average of all Task ID frequency ratings, return result as IWA frequency rating

tasks_iwa_map = tasks_to_dwas[['Task ID','IWA ID']].set_index('Task ID')['IWA ID']

def iwa_task_generator(tasks_to_dwas=tasks_to_dwas, task_matrix=task_matrix):
    """
    Yield IWA and associated tasks within lower, upper date bounds
    """
    ret = []

    idx = 0
    for iwa, group in tasks_to_dwas.groupby('IWA ID'): # over IWAs

        indices = group['Task ID'].values
        #ipdb.set_trace()

        if any(task_matrix.index.isin(indices)): # assumes O(1) hash checking against index
            stats = construct_iwa_statistics(iwa,
                                             task_matrix.loc[indices, :])
            ret.extend(stats)
        else:
            print(iwa, ' has no tasks occuring both before and after {} date'.format(date_cutoff))

        idx += 1
        print("Extracted statistics for iwa {} ({} IWA)".format(iwa, idx))

    df = pd.DataFrame(ret, columns=['IWA ID', 'Category', 'Median Date', 'Median Data Value'])
    return df

def get_mean_iwa_info(iwa, cat, date_cutoff, group, direction='<='):
    df = group.query("Date {} '{}'".format(direction, date_cutoff))

    query = df.query("Category == \'{}\'".format(cat))

    date_mean = query['Date'].astype('int64').mean().astype('datetime64[ns]')
    date = date_mean

    data_value = query['Data Value'].mean()
    return (iwa, cat, date, data_value)

def get_median_iwa_info(iwa, cat, date_cutoff, group, direction='<='):
    df = group.query("Date {} '{}'".format(direction, date_cutoff))

    date = pd.to_datetime('') # null date time, NaT
    data_value = pd.np.nan
    if not df.empty:
        query = df.query("Category == \'{}\'".format(cat))

        if not query.empty:
            sorted_date = query['Date'].sort_values(inplace=False)
            date_median = sorted_date.iloc[ math.ceil(len(sorted_date)/2) - 1 ]
            date = date_median

            data_value = query['Data Value'].median()

    return (iwa, cat, date, data_value)

def construct_iwa_statistics(iwa, group, date_cutoff=date_cutoff):
    """
    Returns sets of 3-tuples of IWA, median date, Category, median Category Date Value
    for before and after the date_cutoff
    """
    ret = []
    categories = [1,2,3,4,5,6,7]

    # for each side of the date cutoff, for each catogory, aggregate task Data Values
    for cat in categories:
        before_values = get_median_iwa_info(iwa, cat, date_cutoff, group)
        after_values = get_median_iwa_info(iwa, cat, date_cutoff, group, direction='>')
        #ipdb.set_trace()

        if not pd.np.isnan( after_values[-1] ): # if we have samples around date_cuttoff. See debug note above
            ret.append( before_values )
            ret.append( after_values )

    return ret

# D) As a final step, we aggregate Rating Data with the task/frequency level rolled up to the IWA level. Most tasks
# point to one IWA but some point to as many as 3, which we average during the task aggreagtion step.

# Notes on Task->DWA->IWA
# see: https://www.onetcenter.org/dictionary/20.1/excel/tasks_to_dwas.html and https://www.onetcenter.org/dictionary/20.1/excel/dwa_reference.html
#
# Essentially each DWA maps to only one IWA (so they're uniquely transitive, don't really mean much)
# Multiple Tasks map to a single IWA/DWA, so we need to group by IWA.
#
# Many tasks make up an IWA, which is why we need to roll up tasks to IWAs.
#
# note: currently the iwa_task_generator() calls the median statistics function, since the average has a bit of skew
#task_matrix['IWA ID'] = task_matrix['Task ID'].map(tasks_iwa_map)

design_matrix = iwa_task_generator()
design_matrix.to_csv(output_matrix_name, sep='\t')
# assert len(design_matrix['IWA ID'].unique()) == 331
