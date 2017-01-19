# Get a list of Intermediate Work Activity descriptions,
# see how closely they associate with 'Artificial Intelligence'

import http.client, urllib.parse
import json
import psycopg2 as pg #postgres db lib
import sys
import time


headers = {
    # Request headers
    'Ocp-Apim-Subscription-Key': '4f8a9145a38a4e6a9327c27ec99db6c5',
}


iwa1 ='Read documents or materials to inform work processes'
iwa2 = 'Study details of artistic productions'
common_terms = ' AND automate'

#Get list of IWAs, IWA ids
#connect to db
try:
    #Get list of IWAS
except Exception as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))

for i in range(0, len(iwas_list)):
    time.sleep(.5) #Limit of 5 api requests per second
    #Format the params for the request
    params = urllib.parse.urlencode({
        # Request parameters
        'q': (iwas_list[i][1] + common_terms),
        'count': '2',
        'offset': '0',
        'mkt': 'en-us',
        'safesearch': 'Moderate',
    })


    try:
        conn = http.client.HTTPSConnection('api.cognitive.microsoft.com')
        conn.request("GET", "/bing/v5.0/search?%s" % params, "{body}", headers)
        response = conn.getresponse()
        data = response.read().decode('utf8')
        jsonData = json.loads(data)
        hits=jsonData['webPages']['totalEstimatedMatches']
        print("{}|{}|{}".format(iwas_list[i][0], iwas_list[i][1], hits))
        conn.close()
    except Exception as e:
        print("[Errno {0}] {1}".format(e.errno, e.strerror))

