# Converts csv file to json
# Props to http://stackoverflow.com/questions/19697846/python-csv-to-json

usage = "Usage: python csv2json.py csvFilename"

import csv #CSV library
import json #JSON library
import sys #to read command line args

#----------------------------------------------------------------------#
# Main functionality                                                   #
#----------------------------------------------------------------------#

#Check arguments for validity
if (len(sys.argv) != 2):
        sys.exit(usage)

#CSV filename should be first argument
csvFilename = sys.argv[1]
if (csvFilename.endswith('.txt') == False):
    sys.exit("Input filename should be in .txt format.")

#Formulate name for json file
jsonFilename = csvFilename.replace('.txt', '.json')

try:
        outfile = open(jsonFilename, 'w')
except:
        sys.exit("Couldn't open json output file")

print>>outfile,"["
        
with open(csvFilename, "r") as fh:
        filereader = csv.reader(fh, delimiter="\t")
        headers = filereader.next()
        for row in filereader:
                newRow = {}
                for i in xrange(0, len(row)):
                        field = row[i]
                        fieldName = headers[i]
                        newRow[fieldName] = field.replace(chr(0xa0), '')
                print>>outfile, ((json.dumps(newRow, indent=4)) + ",")
print>>outfile,"]"
outfile.close()

