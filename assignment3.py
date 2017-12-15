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
3. The path to the file containing the lists of lemmas that are used to construct features
4. The Google API Key; This argument can be ommitted if all entities are already in the
   cache (which is the case for all institution and the place of birth corpus versions)

Example program invocations:

```
python3 assignment3.py lrtrain place-of-birth_train.json place-of-birth_lemmas.json ABc1De2fg3Hi4JkLm567No8p9Qr
```

```
python3 assignment3.py lrclassify 20130403-institution.json institution_lemmas.json
```

Structure of this file:

1. Imports
2. Globals
3. Helper Functions for Main Function
4. Helper Functions for Feature Preparation
5. Main Function
"""

### --- Imports --------------------------------------------------------------------- ###

import io
import json
import numpy as np
import operator
import os
import pickle
import pprint
import re
import spacy
import string
import sys

from IMD_resolver import IMD_resolver
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

### --- Globals --------------------------------------------------------------------- ###

nlp = spacy.load('en')
pp = pprint.PrettyPrinter(indent=4)


### --- Helper Functions for Main Function ------------------------------------------ ###

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
    single letters, conjuctions or articles). These are optional, they can occur in any
    order anm arbitrary number of times between non-secondary tokens (main tokens). At
    least one main token must be in each match, the match must start with a main token
    and it must end with a main token. In between tokens must be at least one whitespace
    or punctuation character.
    
    The function is tolerant to casing (upper/lower case) and to punctuation.
    
    The function returns a list of all found occurrences. For each occurrence the start
    and end positions and the lowercase matched text is returned in a dictionary.
    
    The function has high recall and does OK in terms of precision.
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
    conjunctions = ['and', 'or']
    
    escaped_main_tokens = set()
    escaped_secondary_tokens = set()
    for entity_name_token in entity_name_tokens:
        if len(entity_name_token) > 0:
            if len(entity_name_token) == 1\
                or entity_name_token in all_prepositions\
                or entity_name_token in all_articles\
                or entity_name_token in conjunctions:
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

def store_logistic_regression_object(lr_object, file_name='logistic_regression.pickle'):
    """
    This function stores the `LogisticRegression` object `lr_object` in a file with
    `file_name` using `pickle`.
    
    Thefunction is used to store the `LogisticRegression` object `lr_object` after it has
    been trained with the training corpus so that it can be used for classification by a
    subsequent invocation of the program.
    """
    pickle.dump(lr_object, open(file_name, 'wb'))

def load_logistic_regression_object(file_name='logistic_regression.pickle'):
    """
    This function loads a `LogisticRegression` object from a file with name `file_name`
    using `pickle`.
    
    The function is used to load the `LogisticRegression` object for classification that
    has been trained in a previous invocation of the program.
    """
    return pickle.load(open(file_name, 'rb'))

def collect_lemma_occurrence_counts_as_features(lemma_list, snippet_doc, feature_name_prefix, feature_map, feature_values):
    """
    Collects the occurrence counts of the lemmas in `lemma_list` in the parsed spacy
    document `snippet_doc` as features.
    
    The feature values are added to the `feature_map` and the `feature_values` list
    (structure: see function `collect_features`).
    """
    counts = {}
    
    for lemma in lemma_list:
        counts[lemma] = 0
    
    total = 0
    for token in snippet_doc:
        if token.lemma_ in lemma_list:
            counts[token.lemma_] += 1
            total += 1
    
    for lemma in lemma_list:
        count = counts[lemma]
        value = count / total if total > 0 else 0
        feature_map[feature_name_prefix + lemma] = value
        feature_values.append(value)

def collect_lemma_occurrence_indicators_as_features(lemma_list, snippet_doc, feature_name_prefix, feature_map, feature_values):
    """
    Collects the occurrence indicators (1 if occurs, 0 otherwise) of the lemmas in
    `lemma_list` for the parsed spacy document `snippet_doc` as features.
    
    The feature values are added to the `feature_map` and the `feature_values` list
    (structure: see function `collect_features`).
    """
    indicators = {}
    
    for lemma in lemma_list:
        indicators[lemma] = 0
    
    for token in snippet_doc:
        if token.lemma_ in lemma_list:
            indicators[token.lemma_] = 1
    
    for lemma in lemma_list:
        indicator = indicators[lemma]
        feature_map[feature_name_prefix + lemma] = indicator
        feature_values.append(indicator)

def collect_ner_occurrence_counts_as_features(
    snippet_doc, feature_name_prefix, feature_map, feature_values,
    #entity_types=[
    #    'DATE', 'PERSON'
    #]
    entity_types=[
        'CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY',
        'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME',
        'WORK_OF_ART'
    ]
    ):
    """
    Collects occurrence counts of the `entity_types` in the `snippets_doc` as features.
    
    The feature values are added to the `feature_map` and the `feature_values` list
    (structure: see function `collect_features`).
    """
    counts = {}
    
    for entity_type in entity_types:
        counts[entity_type] = 0
    
    total = 0
    for token in snippet_doc:
        if token.ent_type_ in entity_types:
            counts[token.ent_type_] += 1
            total += 1
    
    for entity_type in entity_types:
        count = counts[entity_type]
        value = count / total if total > 0 else 0
        feature_map[feature_name_prefix + entity_type] = value
        feature_values.append(value)

def collect_ner_occurrence_indicators_as_features(
    snippet_doc, feature_name_prefix, feature_map, feature_values,
    #entity_types=[
    #    'DATE', 'PERSON'
    #]
    entity_types=[
        'CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY',
        'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME',
        'WORK_OF_ART'
    ]
    ):
    """
    Collects occurrence indicators (1|0) of the `entity_types` in the `snippets_doc` as
    features.
    
    The feature values are added to the `feature_map` and the `feature_values` list
    (structure: see function `collect_features`).
    """
    indicators = {}
    
    for entity_type in entity_types:
        indicators[entity_type] = 0
    
    for token in snippet_doc:
        if token.ent_type_ in entity_types:
            indicators[token.ent_type_] = 1
    
    for entity_type in entity_types:
        indicator = indicators[entity_type]
        feature_map[feature_name_prefix + entity_type] = indicator
        feature_values.append(indicator)

def collect_features(subject_name, object_name, subject_name_matches, object_name_matches, item_data, snippet, snippet_doc, lemma_lists):
    """
    Extracts features for a corpus item.
    
    The function returns two values:
    
    - A dictionary mapping the featurne name (some string) to the corresponding feature
      value.
    - A list of the feature values
    """
    # Parse snippet with spacy
    # snippet_doc = nlp(snippet)
    
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
    
    
    # Features derived from lemma `lemma_lists`:
    for (list_index, lemma_list) in enumerate(lemma_lists):
        # FEATURE "lemma_list_count_<i>_<lemma>": The number of occurrences of the
        # lemmas in the current lemma list in the snippet
        collect_lemma_occurrence_counts_as_features(lemma_list, snippet_doc, 'lemma_list_count_{0}_'.format(list_index), feature_map, feature_values)
        
        # FEATURE "lemma_list_indicator_<i>_<lemma>": Indicators for occurrence of the
        # lemmas in the current lemma list in the snippet
        collect_lemma_occurrence_indicators_as_features(lemma_list, snippet_doc, 'lemma_list_indicator_{0}_'.format(list_index), feature_map, feature_values)
    
    # Features derived from named entities found by spacy:
    
    # FEATURE "ner_entity_type_count_<entity_type>": Number of times the entity type
    # occurs in snippet.
    collect_ner_occurrence_counts_as_features(snippet_doc, 'ner_entity_type_count_', feature_map, feature_values)
    
    # FEATURE "ner_entity_type_indicator_<entity_type>": Indicators whether entity type
    # occurs in snippet.
    collect_ner_occurrence_indicators_as_features(snippet_doc, 'ner_entity_type_indicator_', feature_map, feature_values)
    
    
    
    
    subject_as_ne = 0
    subject_as_person = 0
    subject_as_org = 0
    subject_as_gpe = 0
    subject_as_event = 0
    subject_as_loc = 0
    
    for token in snippet_doc:
        for match in subject_name_matches:
            if token.idx >= match['start'] and token.idx < match['end']:
                subject_as_ne = 1
                if token.ent_type_ == 'PERSON':
                    subject_as_person = 1
                elif token.ent_type_ == 'ORG':
                    subject_as_org = 1
                elif token.ent_type_ == 'GPE':
                    subject_as_gpe = 1
                elif token.ent_type_ == 'EVENT':
                    subject_as_event = 1
                elif token.ent_type_ == 'LOC':
                    subject_as_loc = 1
    
    feature_values.append(subject_as_ne)
    # feature_values.append(subject_as_person)
    # feature_values.append(subject_as_org)
    # feature_values.append(subject_as_gpe)
    # feature_values.append(subject_as_event)
    # feature_values.append(subject_as_loc)
        
        #print('- ' + token.orth_ + ' ' + str(token.idx))
        #print('  - ' + snippet[token.idx:min(len(snippet), token.idx + 100)])
    
    
    
    return (feature_map, feature_values)

def format_percents(d):
    """
    Formats a number as percentage (e.g. 0.34234 -> '34.23%')
    """
    return '{:03.2f}'.format(100 * d) + '%'

def print_evaluation(prediction, target_classes, top_margin=False):
    """
    This function prints the evaluation of a prediction (the confusion matrix, the
    baseline and the accuracy of the system).
    """
    if top_margin:
        print('')
    
    sys_pos_tar_pos = 0
    sys_pos_tar_neg = 0
    sys_neg_tar_pos = 0
    sys_neg_tar_neg = 0
    total = len(target_classes)
    
    for (predicted, target) in zip(prediction, target_classes):
        if predicted == 1:
            if target == 1:
                sys_pos_tar_pos += 1
            else:
                sys_pos_tar_neg += 1
        else:
            if target == 1:
                sys_neg_tar_pos += 1
            else:
                sys_neg_tar_neg += 1
    
    print('                  +-----------------+-----------------+-----------------+')
    print('                  | Target positive | Target negative | SUM             |')
    print('+-----------------+-----------------+-----------------+-----------------+')
    print('| System positive | ' + str(sys_pos_tar_pos).rjust(7) + format_percents(sys_pos_tar_pos / total).rjust(8) + ' | ' + str(sys_pos_tar_neg).rjust(7) + format_percents(sys_pos_tar_neg / total).rjust(8) + ' | ' + str(sys_pos_tar_pos + sys_pos_tar_neg).rjust(7) + format_percents((sys_pos_tar_pos + sys_pos_tar_neg) / total).rjust(8) + ' |')
    print('+-----------------+-----------------+-----------------+-----------------+')
    print('| System negative | ' + str(sys_neg_tar_pos).rjust(7) + format_percents(sys_neg_tar_pos / total).rjust(8) + ' | ' + str(sys_neg_tar_neg).rjust(7) + format_percents(sys_neg_tar_neg / total).rjust(8) + ' | ' + str(sys_neg_tar_pos + sys_neg_tar_neg).rjust(7) + format_percents((sys_neg_tar_pos + sys_neg_tar_neg) / total).rjust(8) + ' |')
    print('+-----------------+-----------------+-----------------+-----------------+')
    print('| SUM             | ' + str(sys_pos_tar_pos + sys_neg_tar_pos).rjust(7) + format_percents((sys_pos_tar_pos + sys_neg_tar_pos) / total).rjust(8) + ' | ' + str(sys_pos_tar_neg + sys_neg_tar_neg).rjust(7) + format_percents((sys_pos_tar_neg + sys_neg_tar_neg) / total).rjust(8) + ' | ' + str(total).rjust(7) + '         |')
    print('+-----------------+-----------------+-----------------+-----------------+')
    
    sys_accuracy = (sys_pos_tar_pos + sys_neg_tar_neg) / total
    baseline_accuracy = max((sys_pos_tar_pos + sys_neg_tar_pos) / total, (sys_pos_tar_neg + sys_neg_tar_neg) / total)
    
    print('')
    print('Baseline: ' + format_percents(baseline_accuracy))
    print('System accuracy: ' + format_percents(sys_accuracy))
    
    return (sys_accuracy, baseline_accuracy)

def print_aggregated_evaluation(sys_better_count, avg_sys_accuracy, avg_baseline_accuracy, fold_count, top_margin=False):
    """
    Prints evaluation summary over all folds (how often better than baseline, average
    accuracies)
    """
    baseline_better_count = fold_count - sys_better_count
    
    if top_margin:
        print('')
    print('Summary')
    print('-------')
    print('')
    print('+---------------------------+---------+----------+')
    print('| System better             | ' + str(sys_better_count).rjust(7) + ' | ' +  format_percents(sys_better_count / fold_count).rjust(8) + ' |')
    print('+---------------------------+---------+----------+')
    print('| Baseline better           | ' + str(baseline_better_count).rjust(7) + ' | ' +  format_percents(baseline_better_count / fold_count).rjust(8) + ' |')
    print('+---------------------------+---------+----------+')
    print('| Average accuracy baseline | ' + format_percents(avg_baseline_accuracy).rjust(18) + ' |')
    print('+---------------------------+--------------------+')
    print('| Average accuracy system   | ' + format_percents(avg_sys_accuracy).rjust(18) + ' |')
    print('+---------------------------+--------------------+')

def get_feauter_matrix_and_target_classes_for_corpus_file(corpus_file_path, lemma_lists, api_key, entity_cache, dev_size=None, verbose=False):
    """
    This function processes the corpus file line by line and gets the features and the
    target class for each processable item.
    
    The function relies on other function (e.g. to resolve the entities, to find the
    entities in the snippet, to do the actual feature extraction and to aggregate the
    raters judgments).
    
    It returns the feature matrix (two dimensional list) and the target classes (one
    dimensional list)
    """
    
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
    
    # Useful for debugging, development and testing
    item_counter = -1
    # corpus_items = [] # <- This was used for compilation of k best lemmas (see below)
    
    # These variables are used for storing the first 3000 spacy parsed documents in a
    # cache file (with the name `snippet_docs_pickle_path`). This is because the parsing
    # with spacy takes up a lot of time and causes waiting time during development. This
    # way the code can be very quickly tested with up to 3000 items. We computer refuses
    # to cache all the parses and kills the process when trying to do so.
    processable_item_counter = 0
    snippet_docs = []
    snippet_docs_pickle_path = os.path.splitext(corpus_file_path)[0] + '_docs.pickle'
    
    # Load spacy parses that we might have stored in the cache before.
    if os.path.isfile(snippet_docs_pickle_path):
        snippet_docs = pickle.load(open(snippet_docs_pickle_path, 'rb'))
    
    with open(corpus_file_path, encoding='utf8') as corpus_file:
        for corpus_line in corpus_file:
            item_counter += 1
            
            if verbose:
                print('Processing item ' + '{:06d}'.format(item_counter) + ' (preparation and feature extraction)')
            
            # Useful for debugging and testing. Now that I am using spacy, I can't wait
            # for thousands of items to be processed every time I make a little change in
            # the code.
            if dev_size == 'xs' and item_counter == 100\
                or dev_size == 's' and item_counter == 1000\
                or dev_size == 'm' and item_counter == 3000:
                break
            
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
            
            
            # Get spacy parsed snippet. Add it to cache if it is within the first 3000
            # snippets.
            snippet_doc = None
            if processable_item_counter < len(snippet_docs):
                snippet_doc = snippet_docs[processable_item_counter]
            else:
                snippet_doc = nlp(snippet)
                if len(snippet_docs) < 3000:
                    snippet_docs.append(snippet_doc)
            
            # Collect the features and add the values to the feature matrix
            (feature_map, feature_values) = collect_features(subject_name, object_name, subject_name_matches, object_name_matches, item_data, snippet, snippet_doc, lemma_lists)
            feature_matrix.append(feature_values)
            
            # # `corpus_items` was used to extract lemmas for features (see below). Used
            # # only during feature design. Not used anymore.
            # corpus_items.append(item_data)
            
            processable_item_counter += 1
    
    # # Getting k best lemmas for features (see `get_k_best_lemmas`). Was used for
    # # compiling first lists in institution_lemmas.json and place-of-birth_lemmas.json.
    # # Now that the lists have been created, `get_k_best_lemmas` is not used anymore.
    # 
    
    # Store spacy parses in cache cache
    pickle.dump(snippet_docs, open(snippet_docs_pickle_path, 'wb'))
    
    return (feature_matrix, target_classes)


### --- Helper Functions for Feature Preparation ------------------------------------ ###

def get_k_best_lemmas(corpus_items, k=100, counted_pos=['VERB', 'NOUN', 'ADJ']):
    """
    This function selects the `k` best lemmas occurring in the `corpus_itmes` (best
    features according to skliearn's SelectKBest).
    
    This function was used exclusively for the creating the lists of lemmas that can be
    used to derive features from (occurrence counts). It was applied only to the
    development corpora (neither to the training nor the test corpora). Features are not
    extracted dynamically from training or test corpora.
    
    The `k` best lemmas are returned as a list.
    """
    
    lemmas = set()
    lemma_counts = []
    target = []
    
    for item_data in corpus_items:
        snippet = item_data['evidences'][0]['snippet']
        doc = nlp(snippet)
        
        target.append(is_positive_item(item_data))
        
        item_lemma_counts = {}
        
        for token in doc:
            if not token.like_num and not token.is_punct and not token.is_space and not token.is_bracket and not token.is_quote:
                if token.pos_ in counted_pos:
                    lemmas.add(token.lemma_)
                    if token.lemma_ not in item_lemma_counts:
                        item_lemma_counts[token.lemma_] = 0
                    item_lemma_counts[token.lemma_] += 1
        
        lemma_counts.append(item_lemma_counts)
    
    lemmas = list(lemmas)
    
    vectorizer = DictVectorizer()
    initial_feature_matrix = vectorizer.fit_transform(lemma_counts)
    lemmas = vectorizer.get_feature_names()
    
    k_best = SelectKBest(chi2, k=k)
    
    optimised_feature_matrix = k_best.fit_transform(initial_feature_matrix, target)
    
    selected_lemmas = []
    mask = k_best.get_support()
    
    for selected, lemma in zip(mask, lemmas):
        if selected:
            selected_lemmas.append(lemma)
    
    return selected_lemmas


### --- Main Function --------------------------------------------------------------- ###

def main():
    """
    This is the main function of the script. Using the function defined above it performs
    the tasks required for this programming assignment.
    """
    
    # Get command line attributes
    allowed_actions = ['lrtrain', 'lrclassify', 'lrvalidate']
    if len(sys.argv) < 4 or sys.argv[1] not in allowed_actions:
        print('+----------------------------------------------------------------------+')
        print('| Hello, unfortunately you specified an illegal number of arguments or |')
        print('| requested an unknown action. See in the comment in the source code,  |')
        print('| in the report or below this message how to call the program. See you |')
        print('| soon.                                                                |')
        print('+----------------------------------------------------------------------+')
        print('Usage  : python3 assignment3.py <action: (' + ('|'.join(allowed_actions)) + ')> <path to corpus> <path to lemmas list>[ <Google API Key>]')
        print('Example: python3 ' + allowed_actions[0] + ' assignment3.py place-of-birth_train.json place-of-birth_lemmas.json ABc1De2fg3Hi4JkLm567No8p9Qr')
        print('Example: python3 ' + allowed_actions[1] + ' assignment3.py my_test_corpus.json my_lemmas.json')
        exit(1)
    
    action = sys.argv[1]
    corpus_file_path = sys.argv[2]
    lemma_list_path = sys.argv[3]
    api_key = sys.argv[4] if len(sys.argv) >= 5 else ''
    
    # Load cached entities to minimize time consuming requests
    entity_cache = load_entity_cache()
    
    # Load the lemma lists specified in the command line arguments
    lemma_lists = [[]]
    with open(lemma_list_path, encoding='utf-8') as lemma_lists_file:
        lemma_lists = json.load(lemma_lists_file)
    
    # Get the `feature_matrix` and the (gold) `target_classes`
    (feature_matrix, target_classes) = get_feauter_matrix_and_target_classes_for_corpus_file(corpus_file_path, lemma_lists, api_key, entity_cache, sys.argv[4] if (len(sys.argv) >= 5 and len(sys.argv[4]) <= 2) else None , True)
    
    # Perform the requested action (e.g. training, testing, 10 fold cross validation)
    if action == 'lrtrain':
        logistic_regression = LogisticRegression()
        logistic_regression.fit(feature_matrix, target_classes)
        
        store_logistic_regression_object(logistic_regression)
        
        intercept = logistic_regression.intercept_
        coefficients = logistic_regression.coef_
        
        print('\nLogistic regression:')
        print('- intercept: ' + str(intercept[0]))
        print('- coefficients: ' + ', '.join([str(c) for c in coefficients[0]]))
    elif action == 'lrclassify':
        logistic_regression = load_logistic_regression_object()
        
        prediction = logistic_regression.predict(feature_matrix)
        
        print_evaluation(prediction, target_classes, True)
    else: # lrvalidate
        feature_matrix = np.array(feature_matrix)
        target_classes = np.array(target_classes)
        
        fold_count = 10
        fold_index = 0 # for output
        
        # Variables for aggregated evaluation
        sys_better_count = 0
        sys_accuracy_sum = 0
        baseline_accuracy_sum = 0
        
        k_fold = KFold(n_splits=fold_count, random_state=None, shuffle=False)
        
        for train_index, test_index in k_fold.split(feature_matrix):
            print('\nFold ' + str(fold_index + 1).rjust(2) + ' / ' + str(fold_count).rjust(2))
            print('------------')
            
            train_feature_matrix = feature_matrix[train_index]
            train_target_classes = target_classes[train_index]
            
            test_feature_matrix = feature_matrix[test_index]
            test_target_classes = target_classes[test_index]
            
            logistic_regression = LogisticRegression()
            logistic_regression.fit(train_feature_matrix, train_target_classes)
            
            prediction = logistic_regression.predict(test_feature_matrix)
            
            (sys_accuracy, baseline_accuracy) = print_evaluation(prediction, test_target_classes, True)
            
            sys_accuracy_sum += sys_accuracy
            baseline_accuracy_sum += baseline_accuracy
            if sys_accuracy > baseline_accuracy:
                sys_better_count += 1
            
            fold_index += 1
        
        print_aggregated_evaluation(sys_better_count, sys_accuracy_sum / fold_count, baseline_accuracy_sum / fold_count, fold_count, True)
    
    # Store all found entities in the cache persistent cache
    store_entity_cache(entity_cache)

if __name__ == '__main__':
    main()
