#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
import urllib
import urllib.parse
import urllib.request

def IMD_resolver(entity, api_key, service_url='https://kgsearch.googleapis.com/v1/entities:search'):
    params = {
        'ids': entity,
        'limit': 1,
        'indent': True,
        'key': api_key,
    }
    url = service_url + '?' + urllib.parse.urlencode(params)
    try:
        resource = urllib.request.urlopen(url)
        #response = json.loads(resource.read().decode('utf-8'))
        #entity_name = response['itemListElement'][0]['result']['name']
        # #print(entity_name)
        #return entity_name
        
        # Since downloading all the items takes so much time, we store the entire
        # response for later use. Maybe this helps us later
        response_json = resource.read().decode('utf-8')
        return response_json
    except: # catch *all* exceptions
        print('Error while resolving MID:')
        print(sys.exc_info())
        print(sys.exc_info()[0])
        print(entity)
        return 'False'
