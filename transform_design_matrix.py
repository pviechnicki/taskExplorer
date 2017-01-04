#Transform design matrix from 2 observations and 7 category levels per IWA
# to 1 row per IWA with start value, end value, and independent variables

#----------------------------------------------------------------------#
#Libraries
#----------------------------------------------------------------------#
import csv
import sys
import sqlite3
import os
import re
import getopt

#----------------------------------------------------------------------#
#Globals
#----------------------------------------------------------------------#
iwaFileName = "./design_matrix/design_matrix.csv"
DEBUG_FLAG = False
iwaIdsList = [] #List of IWA id numbers
tempList = []
con = None #db connection
cur = None #db cursor


#----------------------------------------------------------------------#
#Convert a frequency category to hours per year
#----------------------------------------------------------------------#
def frequencyCat2Hours(myFrequencyCategory):
    scaleFactor = .45
    if myFrequencyCategory == 1:
        return .5 * scaleFactor
    elif myFrequencyCategory == 2:
        return 1 * scaleFactor
    elif myFrequencyCategory == 3:
        return 12 * scaleFactor
    elif myFrequencyCategory == 4:
        return 52 * scaleFactor
    elif myFrequencyCategory == 5:
        return 260 * scaleFactor
    elif myFrequencyCategory == 6:
        return 520 * scaleFactor
    elif myFrequencyCategory == 7:
        return 1043 * scaleFactor
    else:
        raise ValueError("Unknown frequency Category passed to frequencyCat2Hours: %s".format(tostring(myFrequencyCategory)))
    return None

#----------------------------------------------------------------------#
#Store a row of the IWA frequency file in temp sqlite table
#----------------------------------------------------------------------#
def storeInTable(myRow):

    global con #sql connection
    global cur #sql cursor
    
    iwa_id = myRow[1]
    frequency_category = int(myRow[2])
    median_date = myRow[3]
    median_data_value = float(myRow[4])
    date_offset = re.search('[0-9]{1,5}', myRow[5]).group(0)

    queryString = "INSERT INTO IWA_FREQ VALUES ('{}', {}, '{}', {}, {})".format(iwa_id, frequency_category, median_date, median_data_value, date_offset)
    
    try:
        cur.execute(queryString)
    except sqlite3.Error as e:
        print("DB insert error {}:".format(e.args[0]))
        return False
    return True

#----------------------------------------------------------------------#
#Initialize sqlite db, create table to store info
#----------------------------------------------------------------------#
def initDB():

    #sql connection and cursor
    global con
    global cur
    
    os.remove('junk.db')
    try:
        con = sqlite3.connect('junk.db')
        cur = con.cursor()
        #Create table
        con.execute('CREATE TABLE IWA_FREQ (IWA_ID TEXT, FREQ_CAT INTEGER, MEDIAN_DATE DATE, MEDIAN_DATA_VALUE REAL, DATE_OFFSET INTEGER)')
        
    except sqlite3.Error as e:
        print("DB initialize error {}:".format(e.args[0]))
        sys.exit(1)
#----------------------------------------------------------------------#
# Print bar-separated headers for output file
#----------------------------------------------------------------------#
def printHeaders():
    headers = ["IWA_ID", "HOURS_PER_YEAR_START", "HOURS_PER_YEAR_END",
               "START_DAYS_OFFSET", "END_DAYS_OFFSET", "DAYS_SPAN"]
    print("|".join(headers))
    

#----------------------------------------------------------------------#
# Retrieve starting and ending values from table for each IWA
#----------------------------------------------------------------------#        
def get_start_value_from_table(myIwaId, myStartOffset):
    global cur
    queryString = "SELECT FREQ_CAT, MEDIAN_DATA_VALUE FROM IWA_FREQ WHERE IWA_ID = '{}' AND DATE_OFFSET = {}".format(myIwaId, myStartOffset)
    answer = 0
    try:
        cur.execute(queryString)
        for row in cur.fetchall():
            if (DEBUG_FLAG):
                print("{}|{}".format(row[0], row[1]))
            answer += frequencyCat2Hours(row[0]) * row[1] * .01
        return answer    
    except sqlite3.Error as e:
        print("DB select error {}:".format(e.args[0]))
    return None
#----------------------------------------------------------------------#        
def get_end_value_from_table(myIwaId, myEndOffset):
    global cur
    queryString = "SELECT FREQ_CAT, MEDIAN_DATA_VALUE FROM IWA_FREQ WHERE IWA_ID = '{}' AND DATE_OFFSET = {}".format(myIwaId, myEndOffset)
    answer = 0
    try:
        cur.execute(queryString)
        for row in cur.fetchall():
            if (DEBUG_FLAG):
                print("{}|{}".format(row[0], row[1]))
            answer += frequencyCat2Hours(row[0]) * row[1] * .01
        return answer    
    except sqlite3.Error as e:
        print("DB select error {}:".format(e.args[0]))
    return None
#----------------------------------------------------------------------#        
def get_start_days_from_table(myIwaId):
    global cur
    queryString = "SELECT DATE_OFFSET FROM IWA_FREQ WHERE IWA_ID = '{}' AND MEDIAN_DATE = (SELECT MIN(MEDIAN_DATE) FROM IWA_FREQ WHERE IWA_ID = '{}')".format(myIwaId, myIwaId)
    try:
        data=cur.execute(queryString)
        return data.fetchone()[0]
    except sqlite3.Error as e:
        print("DB select error {}:".format(e.args[0]))
    return None
#----------------------------------------------------------------------#        
def get_end_days_from_table(myIwaId):
    global cur
    queryString = "SELECT DATE_OFFSET FROM IWA_FREQ WHERE IWA_ID = '{}' AND MEDIAN_DATE = (SELECT MAX(MEDIAN_DATE) FROM IWA_FREQ WHERE IWA_ID = '{}')".format(myIwaId, myIwaId)
    try:
        data=cur.execute(queryString)
        return data.fetchone()[0]
    except sqlite3.Error as e:
        print("DB select error {}:".format(e.args[0]))
    return None

#----------------------------------------------------------------------#        
#Main functionality
#----------------------------------------------------------------------#

def main(argv):
    global DEBUG_FLAG
    global con
    global cur
    global iwaFileName
    
    #Get and parse options
    helpMessage = "Usage: python transform_design_matrix.py -d (DEBUG_MODE) -h (HELP_MODE)"
    
    try:
      opts, args = getopt.getopt(argv,"hd")
    except getopt.GetoptError:
      print(helpMessage)
      sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(helpMessage)
            sys.exit(2)
        elif opt == '-d':
            DEBUG_FLAG = True

    #Initialize sqlite db
    initDB()

    #Read in tab-delimited file
    lineno = 0
    with open(iwaFileName) as f:
        f_csv = csv.reader(f, delimiter='\t')
        for row in f_csv:
            lineno += 1
            if lineno > 1:
                tempList.append(row[1])
                storeInTable(row)
        if (DEBUG_FLAG):
            cur.execute("SELECT COUNT(*) FROM IWA_FREQ")
            tableSize = cur.fetchone()
            sys.stderr.write("Transformed {} lines into {} table entries".format((lineno - 1), tableSize))
        iwaIdsList = set(tempList)
        if (DEBUG_FLAG):
            sys.stderr.write("Found {} IWA ids".format(len(iwaIdsList)))
            
    #for each IWA, calculate the average task hours per year at the beginning
    #and the end
    printHeaders()
    for iwa_id in iwaIdsList:
        #Use the start and end date offsets to help select the data values
        start_days_offset = get_start_days_from_table(iwa_id)
        end_days_offset = get_end_days_from_table(iwa_id)
        days_span = end_days_offset-start_days_offset
        start_value = get_start_value_from_table(iwa_id, start_days_offset)
        end_value = get_end_value_from_table(iwa_id, end_days_offset)
        change = end_value - start_value
        print("{}|{:.2f}|{:.2f}|{:+.2f}|{}|{}|{}".format(iwa_id, start_value,
                                                         end_value,
                                                         change,
                                                         start_days_offset,
                                                         end_days_offset,
                                                         days_span))
        #print out 1 row per IWA, IWA_ID, start_value, end_value, change
        #start_days_offset, end_days_offset, days_span

    #Close db
    if con:
        con.close()

if __name__ == "__main__":
    main(sys.argv[1:])
