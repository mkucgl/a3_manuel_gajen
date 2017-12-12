#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This program is our solution to the programming assignment 3.

It expects the following command line arguments:

1. The action the script should perform. This must be one of:
   
   - "lrtrain": tains the Logistic Regression model; learns the weights from the
     specified training corpus; ("lr" stands for Logistic Regression)
   - "lrclassify": uses the learned weights to classify the items of the specified test
     corpus using the Logistic Regression model.
   - "lrvalidate": performs a 10 fold cross validation of the Logistic Regression model
2. The path to the corpus file
3. The Google API Key; This argument can be ommitted if all entities are already in the
   cache (which is the case for all institution and the place of birth corpus versions).

Example program invocations:

```
python3 assignment3.py lrtrain 20130403-place_of_birth.json ABc1De2fg3Hi4JkLm567No8p9Qr
```

```
python3 assignment3.py lrclassify 20130403-institution.json
```

Structure of this file:

1. Imports
2. Globals
3. Helper Functions for Individual Tasks
4. Main Function
"""

### --- Imports --------------------------------------------------------------------- ###

import io
import json
import re
import spacy
import string
import sys

from IMD_resolver import IMD_resolver
# from nltk import Tree
from sklearn.linear_model import LogisticRegression


### --- Globals --------------------------------------------------------------------- ###

nlp = spacy.load('en')


### --- Helper Functions for Individual Tasks --------------------------------------- ###

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

def find_entity_in_snippet(entity_name, snippet):
    """
    Finds occurrences of an entity name in a snippet.
    
    The function has some tolerance to variations of the `entity_name`, e.g. for
    "Donald J. Trump" it also finds "Trump", "Donald Trump" or "Donald J Trump". Some
    tokens in the `entity_name` are treated as secondary tokens (e.g. prepositions,
    single letters or articles). These are optional, they can occur in any order an
    arbitrary number of times between non-secondary tokens (main tokens). At least one
    main token must be in each match, the match must start with a main token and it must
    end with a main token. In between tokens must be at least one whitespace or
    punctuation character.
    
    The function is tolerant to casing (upper/lower case) and to punctuation.
    
    The function returns a list of all found occurrences. For each occurrence the start
    and end positions and the lowercase matched text is returned in a dictionary.
    """
    
    escaped_punctuation = re.escape(string.punctuation)
    
    # Some normalization before deriving the search regex from the `entity_name`
    # normalised_entity_name = re.sub(punctuation_char_class, '', )
    
    # Split the `normalised_entity_name` into tokens. These will be used to construct the
    # search regex.
    entity_name_tokens = re.split('[%s\s]' % escaped_punctuation, entity_name.lower())
    
    # Construct the regex for searching the entity name (more explanations see
    # description of this function above)
    
    all_prepositions = [
        'aboard', 'about', 'above', 'across', 'after', 'against', 'along', 'amid',
        'among', 'anti', 'around', 'as', 'at', 'before', 'behind', 'below', 'beneath',
        'beside', 'besides', 'between', 'beyond', 'but', 'by', 'concerning',
        'considering', 'despite', 'down', 'during', 'except', 'excepting', 'excluding',
        'following', 'for', 'from', 'in', 'inside', 'into', 'like', 'minus', 'near', 'of',
        'off', 'on', 'onto', 'opposite', 'outside', 'over', 'past', 'per', 'plus',
        'regarding', 'round', 'save', 'since', 'than', 'through', 'to', 'toward',
        'towards', 'under', 'underneath', 'unlike', 'until', 'up', 'upon', 'versus',
        'via', 'with', 'within', 'without'
    ]
    all_articles = ['a', 'an', 'the']
    
    escaped_main_tokens = set()
    escaped_secondary_tokens = set()
    for entity_name_token in entity_name_tokens:
        if len(entity_name_token) > 0:
            if len(entity_name_token) == 1\
                or entity_name_token in all_prepositions\
                or entity_name_token in all_articles:
                escaped_secondary_tokens.add(re.escape(entity_name_token))
            else:
                escaped_main_tokens.add(re.escape(entity_name_token))
    
    if len(escaped_main_tokens) == 0:
        return []
    
    punctuation_or_ws_regex = '[\s' + escaped_punctuation + ']+'
    main_tokens_regex = '(' + ('|'.join(escaped_main_tokens)) + ')+'
    secondary_tokens_regex = '(' + punctuation_or_ws_regex + ('|'.join(escaped_secondary_tokens)) + ')*' if len(escaped_secondary_tokens) > 0 else ''
    
    search_regex = '{0}({1}{2}{0})*'.format(main_tokens_regex, secondary_tokens_regex, punctuation_or_ws_regex)
    
    # Collect all matches and return them as result
    
    match_data = []
    for match in re.finditer(search_regex, snippet.lower()):
        match_data.append({
            'start': match.start(0),
            'end': match.end(0),
            'text': match.group(0)
        })
    
    return match_data

def collect_features(subject_name, object_name, subject_name_matches, object_name_matches, snippet):
    """
    Extracts features for a corpus item.
    
    The function returns two values:
    
    - A dictionary mapping the featurne name (some string) to the corresponding feature
      value.
    - A list of the feature values
    """
    # Dictionary: "feature name -> feature value". Populated below.
    feature_map = {}
    
    # A list just containing the feature values. Populated below.
    feature_values = []
    
    # FEATURE "subject_match_count": Number of `subject_name_matches` (occurrences of the
    # subject entity name in snippet found by `find_entity_in_snippet`)
    subject_match_count = len(subject_name_matches)
    feature_map['subject_match_count'] = subject_match_count
    feature_values.append(subject_match_count)
    
    # FEATURE "object_match_count": Number of `object_name_matches` (occurrences of the
    # object entity name in snippet found by `find_entity_in_snippet`)
    object_match_count = len(object_name_matches)
    feature_map['object_match_count'] = object_match_count
    feature_values.append(object_match_count)
    
    return (feature_map, feature_values)


### --- Main Function --------------------------------------------------------------- ###

def main():
    """
    This is the main function of the script. Using the function defined above it performs
    the tasks required for this programming assignment.
    """
    
    # Get command line attributes
    allowed_actions = ['lrtrain', 'lrclassify', 'lrvalidate']
    if len(sys.argv) < 3 or sys.argv[1] not in allowed_actions:
        print('+----------------------------------------------------------------------+')
        print('| Hello, unfortunately you specified an illegal number of arguments or |')
        print('| requested an unknown action. See in the comment in the source code,  |')
        print('| in the report or below this message how to call the program. See you |')
        print('| soon.                                                                |')
        print('+----------------------------------------------------------------------+')
        print('Usage  : python3 assignment3.py <action: (' + ('|'.join(allowed_actions)) + ')> <path to corpus>[ <Google API Key>]')
        print('Example: python3 ' + allowed_actions[0] + ' assignment3.py 20130403-place_of_birth.json ABc1De2fg3Hi4JkLm567No8p9Qr')
        print('Example: python3 ' + allowed_actions[1] + ' assignment3.py my_test_corpus.json')
        exit(1)
    
    action = sys.argv[1]
    corpus_file_path = sys.argv[2]
    api_key = sys.argv[3] if len(sys.argv) >= 4 else ''
    
    # Load cached entities to minimize time consuming requests
    entity_cache = load_entity_cache()
    
    # Initialize data structures available for holding feature values and gold/target
    # classes
    
    # Two dimensional matrix (a list of feature value for each sample / corpus item).
    # This is the representation `LogisticRegression.fit` expects for the first argument.
    feature_matrix = []
    
    # A "vector" (list) containing the target classes (determined as determined by the
    # function `is_positive_item`). his is the representation `LogisticRegression.fit`
    # expects for the second argument.
    target_classes = []
    
    # Process corpus file line-by-line. Extract features and target classes for all
    # processable corpus items.
    
    item_counter = -1 # Useful for debugging and testing
    with open(corpus_file_path, encoding='utf8') as corpus_file:
        for corpus_line in corpus_file:
            item_counter += 1
            
            # # Useful for debugging and testing. Uncomment if you want to want to try
            # # out code changes on a small subset of the current corpus.
            # if item_counter == 100:
            #    break
            
            # Get item_data: parse corpus line JSON
            item_data = None
            try:
                item_data = json.loads(corpus_line)
            except:
                # A few lines (4) in the institution corpus contain invalid JSON
                # (unescaped backslashes). We ignore these lines.
                continue
            
            # Resolve the entity ids
            
            subject_id = item_data['sub']
            subject_name = get_entity_name(subject_id, api_key, entity_cache)
            
            object_id = item_data['obj']
            object_name = get_entity_name(object_id, api_key, entity_cache)
            
            # As stated in the instructions for resolving entities we can ignore corpus
            # items for which entity resolution fails
            if subject_name is None or object_name is None:
                continue
            
            # Find occurrences of the entity names in the snippet
            
            snippet = item_data['evidences'][0]['snippet']
            
            subject_name_matches = find_entity_in_snippet(subject_name, snippet)
            
            object_name_matches = find_entity_in_snippet(object_name, snippet)
            
            # As stated in the exercise text: if we can not find the entity name in the
            # snippet, the corpus item can be skipped ("If you cannot find the entities
            # in some snippets, you may remove them.")
            if len(subject_name_matches) == 0 or len(object_name_matches) == 0:
                continue
            
            # Determine and remember the target class (used f.e. for training the
            # Logistic Regression model with `LogisticRegression.fit(...)`)
            target_classes.append(1 if is_positive_item(item_data) else 0)
            
            # Collect the features and add the values to the feature matrix
            (feature_map, feature_values) = collect_features(subject_name, object_name, subject_name_matches, object_name_matches, snippet)
            feature_matrix.append(feature_values)
    
    # Perform the requested action (e.g. training, testing, 10 fold cross validation)
    if action == 'lrtrain':
        logistic_regression = LogisticRegression()
        logistic_regression.fit(feature_matrix, target_classes)
        
        intercept = logistic_regression.intercept_
        coefficients = logistic_regression.coef_
        
        # TODO: misclassification error instead of the following output
        print(intercept)
        print(coefficients)
    
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
