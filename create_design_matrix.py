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

import ipdb

# ONET Version specific constants, we
# download those version between LAST/First_ONET_DB
# and change the download url formatting for older versions
db_releases_url = "http://www.onetcenter.org/db_releases.html" # d/l link isn't working!?
OLD_ONET_VERSION = 20.0
LAST_ONET_DB = 21.1
FIRST_ONET_DB = 14.0

output_dir = "./db_releases/"

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

versioned_releases = pd.io.html.read_html(db_releases_url)[0][0]

# Extract, Transform Stages for ONET databaes releases and named tables extracted
named_tables=["Task Ratings.txt", "Task Statements.txt"]
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

merged_table['Date'] = pd.to_datetime(merged_table['Date'], format="%m/%Y")
merged_table.to_csv('merged_'+named_tables[selected_idx], sep='\t')

merged_table = pd.read_csv('merged_'+named_tables[selected_idx], sep='\t')
merged_table['Date'] = pd.to_datetime(merged_table['Date'])

# Analysis Stage

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

date_cutoff = '2010-01-01'
after_cutoff = unique_tasks_by_date['Date'] > date_cutoff

# filter out those rows not having a second update (see: after_mask) occuring after 2010-01-01 (see: after_cutoff)
all_tasks = merged_table['Task ID'].isin( unique_tasks_by_date.loc[after_mask & after_cutoff, 'Task ID'] )

# (... gives 11,096 tasks (out of 20,200, e.g. merged_table['Task ID'].unique()))
# ... add a column for time differential from task start date (in years)

# todo: fix up this logic to create a date map for task ids
unique_tasks_by_date.loc[before_mask, ['Date', 'Task ID']].set_index('Task ID')
