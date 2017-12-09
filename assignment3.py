#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import json
import sys

from IMD_resolver import IMD_resolver

"""
This is file contains the program that solving the tasks in the programming assignemt 1.

It expects 2 command line parameters:

1. The path to the corpus file
2. The Google API Key

Example program invocations: 

```
python3 assignment3.py 20130403-place_of_birth.json ABc1De2fg3Hi4JkLm567No8p9QrsTuVwABc1De2
python3 assignment3.py 20130403-institution.json ABc1De2fg3Hi4JkLm567No8p9QrsTuVwABc1De2
```

"""

def load_entity_cache(cache_file='entity_cache.json'):
    """
    This function loads the cached IMD resolution responses from the `chache_file` into a
    dictionary and returns this dictionary.
    """
    
    entity_cache = None
    
    with open(cache_file, encoding='utf-8') as file:
        entity_cache = json.load(file)
    
    return entity_cache

def store_entity_cache(entity_cache, cache_file='entity_cache.json'):
    """
    This function stores the entity cache from memory (dictionary `entity_cache`) in the
    `cache_file`.
    """
    
    with io.open(cache_file, 'w', encoding='utf8') as file:
        json.dump(entity_cache, file, ensure_ascii=False)

def get_entity_name(entity_id, api_key, entity_cache):
    """
    Resolves an entity id. If the data is found in the `entity_cache`, the cached result
    is returned. Otherwise a request to the API is made and the retrieved data is added
    to the `entitiy_cache`.
    
    `None` is returned if request failed or a failed request is cached for èntity_id`.
    """
    
    response_data = None
    
    if entity_id in entity_cache:
        response_data = entity_cache[entity_id]
    else:
        response_json = IMD_resolver(entity_id, api_key)
        if response_json != False and response_json != 'False':
            parsed_response = json.loads(response_json)
            if 'itemListElement' in parsed_response and len(parsed_response['itemListElement']) > 0 and 'result' in parsed_response['itemListElement'][0] and 'name' in parsed_response['itemListElement'][0]['result']:
                response_data = parsed_response
                entity_cache[entity_id] = response_data
            else:
                entity_cache[entity_id] = None
        else:
            entity_cache[entity_id] = None
    
    if response_data is None:
        return None
    
    return response_data['itemListElement'][0]['result']['name']

def is_positive_item(item_data, agreement_threshold=0.5):
    """
    Decides for an item (`item_data`: parsed json line from the corpus) whether it is a
    positive example of the relation. An item is treated as a positive example if the
    percentage of judges agreeing is at least `agreement_threshold`.
    """
    
    agreeing_count = 0
    for judgment_data in item_data['judgments']:
        if judgment_data['judgment'] == 'yes':
            agreeing_count += 1
    
    judgment_count = len(item_data['judgments'])
    judgment_share = agreeing_count / judgment_count if judgment_count > 0 else 0
    
    return judgment_share >= agreement_threshold

def main():
    """
    This is the main function of the script. Using the function defined above it performs
    the tasks required for this programming assignment.
    """
    
    # Get command line attributes
    if len(sys.argv) != 3:
        print('Usage  : python3 assignment3.py <path to corpus> <Google API Key>')
        print('Example: python3 assignment3.py 20130403-place_of_birth.json ABc1De2fg3Hi4JkLm567No8p9QrsTuVwABc1De2')
        exit(1)
    
    corpus_file_path = sys.argv[1]
    api_key = sys.argv[2]
    
    # Load cached entities to minimize time consuming requests
    entity_cache = load_entity_cache()
    
    # Process corpus file line-by-line
    with open(corpus_file_path, encoding='utf8') as corpus_file:
        for corpus_line in corpus_file:
            # Get item_data: parse corpus line JSON
            item_data = None
            try:
                item_data = json.loads(corpus_line)
            except:
                # A few lines (4) in the institution corpus contain invalid JSON
                # (unescaped backslashes). We ignore these lines.
                continue
            
            # is_positive_item(item_data)
            
            subject_id = item_data['sub']
            subject_name = get_entity_name(subject_id, api_key, entity_cache)
            
            object_id = item_data['obj']
            object_name = get_entity_name(object_id, api_key, entity_cache)
            
            
            print(subject_name)
            print(object_name)
            
            # As stated in the instructions for resolving entities we can ignore corpus
            # items for which entity resolution fails
            if subject_name is None or object_name is None:
                continue
    
    
    # Store all found entities in the cache persistent cache
    store_entity_cache(entity_cache)
    
    
    # -----------------------------------------------------------------------------------
    # The following is test code that was used during development. We leave it here in
    # in case you are interested in our development process.
    # -----------------------------------------------------------------------------------
    
    # # An initial test whether the import of the IMD_resolver and the creation of the
    # # API key worked.
    # test_entity = '/m/0dl567'
    # test_entity_name = IMD_resolver(test_entity, api_key, service_url)
    # print(test_entity_name)
    
    # # A test with the modified IMD_resolver that returns the entire response JSON
    # test_entity = '/m/0dl567'
    # test_entity_response_json = IMD_resolver(test_entity, api_key, service_url)
    # print(test_entity_response_json)
    
    # # Testint entity cache roundtrip
    # entity_cache = { 'x' : 'abc' }
    # store_entity_cache(entity_cache)
    # loaded_entity_cache = load_entity_cache()
    # print(entity_cache)
    
    # # Test retrieving entity name with cache
    # test_entity = '/m/0dl567'
    # test_entity_name = get_entity_name(test_entity, api_key, entity_cache)
    # print(test_entity_name)

if __name__ == '__main__':
    main()
