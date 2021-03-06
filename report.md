Programming Assignment 3
========================

Handed in by Manuel Kunz and Gajendira Sivajothi

Introduction
------------

The code we produced to solve the programming assignment can be found in the file
"assignment3.py". We created a single classifier that can be used for both types of data
(the path to a list of institution or place of birth specific lemmas can be specified via
the command line arguments described just below). This one program can perform the
training (action `lrtrain`), the testing/classification (action `lrclassify`) and the
10-fold cross-validation (action `lrvalidate`).

Note: After the program has trained (after an invocation with the action `lrtrain`), it
stores the trained classifier for later use with the action `lrclassify`.

This is how you call the program (examples below):

```
python3 assignment3.py <action: lrtrain|lrclassify|lrvalidate> <path to corpus> <path to lemmas list>[ <Google API Key>]
```

It expects the following command line arguments:

1. The action the script should perform. This must be one of:
   
   - `lrtrain`: tains the Logistic Regression model; learns the weights from the
     specified training corpus; ("lr" stands for Logistic Regression)
   - `rclassify`: uses the learned weights to classify the items of the specified test
     corpus items (using the trained Logistic Regression model stored during the
     execution of the `lrtrain` action).
   - `lrvalidate`: performs a 10 fold cross validation
2. The path to the corpus file (e.g. "institution_test.json")
3. The path to the file containing the argument lists from which features are constructed
   (e.g. "institution_args.json" or "place-of-birth_args.json")
4. The Google API Key; This argument can be ommitted if all entities are already in the
   cache (which is the case for all institution and the place of birth data set versions)

Example 1 (only training; use your own API key if new features need to be downloaded,
otherwise you can omit the API key):

```
python3 assignment3.py lrtrain institution_train.json institution_args.json G00gleAPIk3y
```

Example 2 (training, then testing/classification):

```
python3 assignment3.py lrtrain institution_train.json institution_args.json
python3 assignment3.py lrclassify institution_test.json institution_args.json
```

Example 3 (10-fold cross validation):

```
python3 assignment3.py lrvalidate place-of-birth_nodev.json place-of-birth_args.json
```

Documentation of Development
----------------------------

In this section we describe the development process that lead to the solution we handed
in. The actual code can be found in the file "assignment3". It is well structured into
functions and all functions are extensively commented. In order to keep a clean structure
for this report we did not copy/paste the code sections into the report. Instead we
reference the relevant function names and the reader can look up the implementation
details in the code.

### Exercise 1 and General Preparation

1.  After reading the entire exercise sheet we created the file "assignment3.py" for our
    solution. We decided to create a single program that can classify both types of
    relations. An empty `main` function was created.
2.  We downloaded the files "20130403-institution.json" and
    "20130403-place-of-birth.json"
3.  We inspected the instructions/examples on how to resolve IDs (ResolvingIDs.docx). We
    assessed the requirements for using ID resolution and created a Google account for
    our team and an API key for this account.
4.  We created the file "IMD_resolver.py" which initially contained the exact code found
    in "ResolvingIDs.docx" and included it in "assignment3.py". We added all the
    necessary `import`s in both files and tries to resolve some first IDs. After this
    worked made a small change to "IMD_resolver.py" so that it returnes the entire data
    retrieved from Google (and not just the entity name) so that we could later store all
    the information locally (because we had heard that downloading all the entities takes
    a long time and we did not want to risk doing it twice).
    
    Now we were able to resolve IDs (without local cache).
5.  Since downloading all the entity data really takes a long time, we then implemented
    local caching. When new entities are downloaded, they are added to a dictionary
    during runtime (in code `entity_cache`). After programm execution the script should
    store all entity data it knows in a local file (file: "entity_cache.json", see
    function: `store_entity_cache`). When the program starts it should load all the
    entity data gathered in earlier executions (function: `load_entity_cache`).
    
    For convenient use we wrote the function `get_entity_name` which gets the entity name
    from the `entity_cache` if possible and defaults to downloading the entity data if
    necessary.
6.  We put the code for processing a data set file line-by-line into "assignment3.py",
    parsed the json for each line (reacted to illegal backslashes) and called
    `get_entity_name` on all subject/object IDs. This way we built up our cache with all
    the necessary entity data (which took a whole night to execute because this one time
    all the data had to be downloaded from Google).
7.  Next we addressed the problem of aggregating the raters' judgments into a single
    judgment (multiple yes/no into a single yes/no): the target/gold class. We chose to
    treat an item positively (relation present) if a certain percentage of raters said
    yes (default 50%). For this we implemented the function `is_positive_item`.
8.  Next we tackled the problem of finding the entity names in the snippet. After looking
    at some of the entity names and the respective snippets (`print`), we chose to go
    with a regular expression approach to finding the entity names. This is implemented
    in the function `find_entity_in_snippet` which has a comment explaining how it works
    in detail.
9.  Then we split the corpora/data sets (development/training/test). The exact way the
    sets were split is described below (see section [Data Sets](#data-sets) below).
    
    The "_nodev" data sets (also described in [Data Sets](#data-sets) below) containing
    all but the development data on which the 10-fold cross-validation could be run were
    created only later during the implementation of the 10-fold cross validation (see
    below).

### Exercise 2

1.  We created the function `collect_features` for extracting features.
2.  We inserted two very simple features: `subject_match_count` and `object_match_count`
    (features are described below) to have some features to work with during development.
3.  At this point we did not yet know how the Logistic Regression had to be performed
    (What has to be `import`ed/installed? What arguments will the package expect? What
    results will it produce?).
    
    - We did not know how much time it would take to implement the use of the Logistic
      Regression.
    - We wanted to know in which format the features had to be represented.
    - We wanted to see what result the features ultimately produce.
    
    Therefore, instead of adding more features right away, we continued with the
    implementation of the code for the call to the Logistic Regression (described below
    in the next section [Exercise 3](#exercise-3)).
4.  Since we wanted to use some lemmas as features (occurrence indicators), we tried to
    produce a lemma list automatically with sklearn's `SelectKBest` from the development
    data sets. For this the function `get_k_best_lemmas` was implemented. The result were
    the JSON list files "institution_args.json" and "place-of-birth_args.json". We
    implemented the functionality of accepting a list file as command line parameter
    (in `main` and in `get_feature_matrix_and_target_classes_for_corpus_file`) and
    derived the feature values in `collect_features`.
5.  Since we have started using spaCy and noticed that it takes a lot of time parsing
    each snippet (`nlp(snippet)` for thousands of snippets), we implemented a mechanism
    that stores the spaCy parses between executions of the program and loads them again
    (`pickle.load(open(snippet_docs_pickle_path, 'rb'))`,
    `pickle.dump(snippet_docs, open(snippet_docs_pickle_path, 'wb'))`) for the first
    3000 items. This way a lot of waiting could be avoided.
6.  Finally we added features until the time for this programming assignment was up. All
    the features are described below and in the function `collect_features`.

### Exercise 3

1.  We researched sklearn's `LogisticRegression` and made a simple test with dummy data
    with the `feature_matrix` initially containing only two columns (see above). The
    `fit` and `predict` methods worked and we were able to retrieve the coefficients 
    (weights) we expected from the lecture.
2.  Our program was structured to have separate training (`lrtrain`) and testing
    (`lrclassify`) actions which which would be run individually/consecutively. To make
    this work we had to store the trained weights between executions. There seems to be
    no way of creating a `LogisticRegression` object only from weights. We therefore
    stored the entire object using `pickle` (see functions
    `store_logistic_regression_object` and `load_logistic_regression_object`). This works
    very well.
3.  We collected the first evaluation data (initially only system and baseline accuracy).
    We chose the following [baseline](#baseline): The baseline performance is the highest
    performance achievable assigning the same class to all the items (assigning the class
    that is more frequent in the data set). We printed this initial evaluation data and
    were "shocked" to see that with the initial 2 features our classifier exactly follows
    the baseline strategy. Of course this changed when we started adding more features
    but until the end our classifier had a tendency of classifying too many items as
    positive (see also [Error Analysis](#error-analysis)).
4.  We implemented the 10-fold cross-validation using sklearn's `KFold` (see `main`
    function under action `lrvalidate` in "assignment3.py"). As mentioned above, we
    created the "_nodev" data sets (described below) for running the 10-fold
    cross-validation on.
5.  We refined our evaluation code especially to also include a confusion matrix and
    average performance data over the folds (used for the `lrclassify` and the
    `lrvalidate` actions). The function where you can find the details is
    `print_evaluation`. To see the results you can run the program or look at the results
    provided in [Appendix 1: Results](#appendix-1-results).

### Exercise 4

1.  In the very end we wrote this report.

### Exercise 5

1.  Looking forward to the presentation next Tuesday.

Decisions
---------

### Data Sets

Each of the downloaded corpora we split into three parts:

- Development data set (5% of the lines): Used for inspection during development /
  feature design. Files containing these data sets are suffixed with "_dev".
- Training data set (75% of the lines): Used for training the Logistic Regression
  classifier. Files containing these data sets are suffixed with "_train".
- Test data set (20% of the lines): Used for testing the Logistic Regression classifier
  trained with the training set. Files containing these data sets are suffixed with
  "_test".

For the development data set for example we took the top most 5% of the lines. The
following 75% of the lines were assigned to the training set. The last 20% make up the
test set.

In addition we created files that contain all the lines except for those used in the
development data sets. These have the name suffix "_nodev". They were used to run the
10-fold cross-validation (action `lrvalidate`). This is how the results at the end of the
report were created.

### Aggregation of Judgments

We chose to treat an item as positive (relation present) if at least 50% of the raters
said "yes" to it.

### Baseline

As a baseline we chose the strategy of only assigning one class. The baseline performance
is the best performance that could be achieved using this strategy (always assigning the
class that occurs more often in the data).

### Features

The code for extracting the features can be found in the function `collect_features`.

#### Subject / Object Name in Snippet

The intuition behind these features is to allow the classifier to figure out whether the
entities identified by the IDs really are mentioned in the snippet (and not for example
some other person with the same name or some other city in a different state / country).

- `subject_match_count`: Number of `subject_name_matches` (occurrences of the subject
  entity name in snippet found by the function `find_entity_in_snippet`)
- `object_match_count`: Number of `object_name_matches` (occurrences of the object
  entity name in snippet found by the function `find_entity_in_snippet`)
- `exact_subject_name_match_indicator`: Indicates (1: yes, 0: no) whether the exact
  subject name retrieved from Google was found in the snippet
- `exact_object_name_match_indicator`: Indicates (1: yes, 0: no) whether the exact object
  name retrieved from Google was found in the snippet
- `object_comma_appended_place_in_snippet`: Indicates (1: yes, 0: no) whether the upper
  case part of the object name after the comma (that often is a country or state name) is
  found in the snippet
- `<all|first>_subject_name_parts_in_snippet`: Indicates (1: yes, 0: no) whether the
  first|all (non-1-length, whitespace or punctuation separated) parts of
  the subject name are found in the snippet
- `all_object_name_parts_in_snippet`: Indicates (1: yes, 0: no) whether all
  (non-1-length, whitespace or punctuation separated) parts of the object name are found
  in the snippet
- `all_upper_case_object_name_parts_in_snippet`: Indicates (1: yes, 0: no) whether all
  (non-1-length, whitespace or punctuation separated) upper case parts of the object
  name are found in the snippet

#### Lemmas

These features are constructed from the lemma list read from the file provided as third
command line parameter. The intuition is that the words used in the snippets give a clue
whether they describe the relationship in question. Whether or not a specific lemma
occurs in the snippet might therefor help the classifier make its judgment. Playing
around with the code these features have proven to have a great effect on the accuracy
of the classifier.

- `lemma_list_count_<lemma>`: The number of occurrences of the lemmas in the current
  lemma list in the snippet (normalized by the total number of lemmas).
- `lemma_list_indicator_<lemma>`: Indicators for occurrence of the lemmas in the list in
  the snippet

#### SpaCy Named Entities

These features try to provide some of the named entity information provided by SpaCy to
the classifier.

- `ner_entity_type_count_<entity_type>`: Number of times the entity type occurs in
  snippet
- `ner_entity_type_indicator_<entity_type>`: Indicators whether entity type occurs in
  snippet
- `subject_as_ne`: Value that indicates (1: yes, 0: no) whether a named entity was found
  by spaCy overlapping a subject name match.
- `subject_as_entity_type_<entity_type>_indicator`: Value that indicates whether a
  subject name match intersects a named entity token of type `<entity_type>` identified
  by spaCy. 1 if spaCy found a named entity of type `<entity_type>` intersecting a
  subject name match, 0 otherwise.
- `subject_as_entity_type_<entity_type>_count`: Number of spaCy tokens of named entity
  type `<entity_type>` overlapping a subject name match.
- `subject_as_entity_type_<entity_type>_share`: The percentage of all named entity tokens
  found by spaCy overlapping a subject name match that are of type `<entity_type>`.
- `object_as_ne`: Value that indicates (1: yes, 0: no) whether a named entity was found
  by spaCy overlapping a object name match.
- `object_as_entity_type_<entity_type>_indicator`: Value that indicates whether an object
  name match intersects a named entity token of type `<entity_type>` identified by spaCy.
  1 if spaCy found a named entity of type `<entity_type>` intersecting an object name
  match, 0 otherwise.
- `object_as_entity_type_<entity_type>_count`: Number of spaCy tokens of named entity
  type `<entity_type>` overlapping an object name match.
- `object_as_entity_type_<entity_type>_share`: The percentage of all named entity tokens
  found by spaCy overlapping an object name match that are of type `<entity_type>`.

#### Personal Pronouns

Since they are so common we thought they might bear some information.

- `he_indicator`: Value (1: yes, 0: no) indicating whether the snippet contains the
  personal pronoun "he".
- `she_indicator`: Value (1: yes, 0: no) indicating whether the snippet contains the
  personal pronoun "she".

#### Sentence Structure and Dependency Tree

We thought the properties of the sentences and the position of the subject/object
tokens/matches therein might give a hint at the content. The dependency tree was
mentioned in articles as possible feature.

- `subject_name_in_first_sentence`: Indicates (1: yes, 0: no) whether the subject name
  occurred in the first sentence
- `object_name_in_first_sentence`: Indicates (1: yes, 0: no) whether the object name
  occurred in the first sentence
- `first_sentence_with_subject`: Index of the first sentence in which the subject name
  was found
- `first_sentence_with_object`: Index of the first sentence in which the object name was
  found
- `subject_and_object_in_same_sentence`: Indicates (1: yes, 0: no) whether the subject
  and object name have ever been found in the same sentence
- `avg_sentence_length`: The average sentence length measured in spaCy tokens.
- `<shortest|longest|first>_sentence_length`: Length of the shortest| longest|first
  sentence measured in spaCy tokens
- `sentence_counter`: The number of sentences in the snippet
- `subject_object_min_distance`: Minimum number of tokens between occurrences
- `subj_dep_indicator_<relation_type>`: Indicates (1: yes, 0: no) whether a subject token
  ever occurs in a (spaCy) dependency relation of type `<relation_type>` to its head.
- `obj_dep_indicator_<relation_type>`: Indicates (1: yes, 0: no) whether a object token
  ever occurs in a (spaCy) dependency relation of type `<relation_type>` to its head.
- `<min|max>_<subj|obj>_root_path_len`: Minimal|maximal length of the path from a
  subject|object token to the root of the spaCy dependency tree it belongs to

Error Analysis
--------------

Most of the errors the classifier makes at the moment are false positives (it classifies
the snippet as containing the relation even though the rater's majority said "no").

Even for us as humans it is often not easy to understand why some snippets are judged
negatively. Many of them contain the wrong person or contain slightly ambiguous
statements (e.g. saying that someone was born *near* a place or received a degree from
some institution but then continued their carrer at an other institution or in an other
field). Concentrating more on trying to identify whether the right
person/place/institution features in the snippet seems a good strategy for further
development of the classifier (as opposed to trying to identify whether the snippet
contains the relation for some subject/object pair, not necessarily those with the Google
API IDs from the data).

The lemma lists turned out to have a large effect on accuracy. We realized that
additional features can even decrease accuracy (overfitting?). That in mind the
classifier could maybe improved in a future development effort by fiddling around more
with these lemma lists (especially dropping some of the more suspicious lemmas).

Finally we realized that, as suggested in the exercise, the percentage of raters needed
to treat an item as positive (target/gold class) has a large effect on the performance
of the classifier.

Appendix 1: Results
-------------------

```
$ python3 assignment3.py lrvalidate institution_nodev.json institution_args.json
Processing item 000000 (preparation and feature extraction)
Processing item 000001 (preparation and feature extraction)
Processing item 000002 (preparation and feature extraction)
...
Processing item 040494 (preparation and feature extraction)
Processing item 040495 (preparation and feature extraction)
Processing item 040496 (preparation and feature extraction)

Fold  1 / 10
------------

                  +-----------------+-----------------+-----------------+
                  | Target positive | Target negative | SUM             |
+-----------------+-----------------+-----------------+-----------------+
| System positive |    2047  65.46% |     600  19.19% |    2647  84.65% |
+-----------------+-----------------+-----------------+-----------------+
| System negative |     148   4.73% |     332  10.62% |     480  15.35% |
+-----------------+-----------------+-----------------+-----------------+
| SUM             |    2195  70.20% |     932  29.80% |    3127         |
+-----------------+-----------------+-----------------+-----------------+

Baseline: 70.20%
System accuracy: 76.08%

Fold  2 / 10
------------

                  +-----------------+-----------------+-----------------+
                  | Target positive | Target negative | SUM             |
+-----------------+-----------------+-----------------+-----------------+
| System positive |    2040  65.24% |     614  19.64% |    2654  84.87% |
+-----------------+-----------------+-----------------+-----------------+
| System negative |     146   4.67% |     327  10.46% |     473  15.13% |
+-----------------+-----------------+-----------------+-----------------+
| SUM             |    2186  69.91% |     941  30.09% |    3127         |
+-----------------+-----------------+-----------------+-----------------+

Baseline: 69.91%
System accuracy: 75.70%

Fold  3 / 10
------------

                  +-----------------+-----------------+-----------------+
                  | Target positive | Target negative | SUM             |
+-----------------+-----------------+-----------------+-----------------+
| System positive |    2062  65.94% |     577  18.45% |    2639  84.39% |
+-----------------+-----------------+-----------------+-----------------+
| System negative |     161   5.15% |     327  10.46% |     488  15.61% |
+-----------------+-----------------+-----------------+-----------------+
| SUM             |    2223  71.09% |     904  28.91% |    3127         |
+-----------------+-----------------+-----------------+-----------------+

Baseline: 71.09%
System accuracy: 76.40%

Fold  4 / 10
------------

                  +-----------------+-----------------+-----------------+
                  | Target positive | Target negative | SUM             |
+-----------------+-----------------+-----------------+-----------------+
| System positive |    2068  66.13% |     606  19.38% |    2674  85.51% |
+-----------------+-----------------+-----------------+-----------------+
| System negative |     157   5.02% |     296   9.47% |     453  14.49% |
+-----------------+-----------------+-----------------+-----------------+
| SUM             |    2225  71.15% |     902  28.85% |    3127         |
+-----------------+-----------------+-----------------+-----------------+

Baseline: 71.15%
System accuracy: 75.60%

Fold  5 / 10
------------

                  +-----------------+-----------------+-----------------+
                  | Target positive | Target negative | SUM             |
+-----------------+-----------------+-----------------+-----------------+
| System positive |    2087  66.74% |     567  18.13% |    2654  84.87% |
+-----------------+-----------------+-----------------+-----------------+
| System negative |     152   4.86% |     321  10.27% |     473  15.13% |
+-----------------+-----------------+-----------------+-----------------+
| SUM             |    2239  71.60% |     888  28.40% |    3127         |
+-----------------+-----------------+-----------------+-----------------+

Baseline: 71.60%
System accuracy: 77.01%

Fold  6 / 10
------------

                  +-----------------+-----------------+-----------------+
                  | Target positive | Target negative | SUM             |
+-----------------+-----------------+-----------------+-----------------+
| System positive |    2033  65.01% |     605  19.35% |    2638  84.36% |
+-----------------+-----------------+-----------------+-----------------+
| System negative |     142   4.54% |     347  11.10% |     489  15.64% |
+-----------------+-----------------+-----------------+-----------------+
| SUM             |    2175  69.56% |     952  30.44% |    3127         |
+-----------------+-----------------+-----------------+-----------------+

Baseline: 69.56%
System accuracy: 76.11%

Fold  7 / 10
------------

                  +-----------------+-----------------+-----------------+
                  | Target positive | Target negative | SUM             |
+-----------------+-----------------+-----------------+-----------------+
| System positive |    2083  66.63% |     564  18.04% |    2647  84.68% |
+-----------------+-----------------+-----------------+-----------------+
| System negative |     155   4.96% |     324  10.36% |     479  15.32% |
+-----------------+-----------------+-----------------+-----------------+
| SUM             |    2238  71.59% |     888  28.41% |    3126         |
+-----------------+-----------------+-----------------+-----------------+

Baseline: 71.59%
System accuracy: 77.00%

Fold  8 / 10
------------

                  +-----------------+-----------------+-----------------+
                  | Target positive | Target negative | SUM             |
+-----------------+-----------------+-----------------+-----------------+
| System positive |    2058  65.83% |     616  19.71% |    2674  85.54% |
+-----------------+-----------------+-----------------+-----------------+
| System negative |     158   5.05% |     294   9.40% |     452  14.46% |
+-----------------+-----------------+-----------------+-----------------+
| SUM             |    2216  70.89% |     910  29.11% |    3126         |
+-----------------+-----------------+-----------------+-----------------+

Baseline: 70.89%
System accuracy: 75.24%

Fold  9 / 10
------------

                  +-----------------+-----------------+-----------------+
                  | Target positive | Target negative | SUM             |
+-----------------+-----------------+-----------------+-----------------+
| System positive |    2043  65.36% |     611  19.55% |    2654  84.90% |
+-----------------+-----------------+-----------------+-----------------+
| System negative |     170   5.44% |     302   9.66% |     472  15.10% |
+-----------------+-----------------+-----------------+-----------------+
| SUM             |    2213  70.79% |     913  29.21% |    3126         |
+-----------------+-----------------+-----------------+-----------------+

Baseline: 70.79%
System accuracy: 75.02%

Fold 10 / 10
------------

                  +-----------------+-----------------+-----------------+
                  | Target positive | Target negative | SUM             |
+-----------------+-----------------+-----------------+-----------------+
| System positive |    1990  63.66% |     649  20.76% |    2639  84.42% |
+-----------------+-----------------+-----------------+-----------------+
| System negative |     157   5.02% |     330  10.56% |     487  15.58% |
+-----------------+-----------------+-----------------+-----------------+
| SUM             |    2147  68.68% |     979  31.32% |    3126         |
+-----------------+-----------------+-----------------+-----------------+

Baseline: 68.68%
System accuracy: 74.22%

Summary
-------

+---------------------------+---------+----------+
| System better             |      10 |  100.00% |
+---------------------------+---------+----------+
| Baseline better           |       0 |    0.00% |
+---------------------------+---------+----------+
| Average accuracy baseline |             70.55% |
+---------------------------+--------------------+
| Average accuracy system   |             75.84% |
+---------------------------+--------------------+


$ python3 assignment3.py lrvalidate place-of-birth_nodev.json place-of-birth_args.json
Processing item 000000 (preparation and feature extraction)
Processing item 000001 (preparation and feature extraction)
Processing item 000002 (preparation and feature extraction)
...
Processing item 009086 (preparation and feature extraction)
Processing item 009087 (preparation and feature extraction)
Processing item 009088 (preparation and feature extraction)

Fold  1 / 10
------------

                  +-----------------+-----------------+-----------------+
                  | Target positive | Target negative | SUM             |
+-----------------+-----------------+-----------------+-----------------+
| System positive |     578  69.81% |     127  15.34% |     705  85.14% |
+-----------------+-----------------+-----------------+-----------------+
| System negative |      33   3.99% |      90  10.87% |     123  14.86% |
+-----------------+-----------------+-----------------+-----------------+
| SUM             |     611  73.79% |     217  26.21% |     828         |
+-----------------+-----------------+-----------------+-----------------+

Baseline: 73.79%
System accuracy: 80.68%

Fold  2 / 10
------------

                  +-----------------+-----------------+-----------------+
                  | Target positive | Target negative | SUM             |
+-----------------+-----------------+-----------------+-----------------+
| System positive |     568  68.60% |     128  15.46% |     696  84.06% |
+-----------------+-----------------+-----------------+-----------------+
| System negative |      42   5.07% |      90  10.87% |     132  15.94% |
+-----------------+-----------------+-----------------+-----------------+
| SUM             |     610  73.67% |     218  26.33% |     828         |
+-----------------+-----------------+-----------------+-----------------+

Baseline: 73.67%
System accuracy: 79.47%

Fold  3 / 10
------------

                  +-----------------+-----------------+-----------------+
                  | Target positive | Target negative | SUM             |
+-----------------+-----------------+-----------------+-----------------+
| System positive |     577  69.69% |     109  13.16% |     686  82.85% |
+-----------------+-----------------+-----------------+-----------------+
| System negative |      46   5.56% |      96  11.59% |     142  17.15% |
+-----------------+-----------------+-----------------+-----------------+
| SUM             |     623  75.24% |     205  24.76% |     828         |
+-----------------+-----------------+-----------------+-----------------+

Baseline: 75.24%
System accuracy: 81.28%

Fold  4 / 10
------------

                  +-----------------+-----------------+-----------------+
                  | Target positive | Target negative | SUM             |
+-----------------+-----------------+-----------------+-----------------+
| System positive |     570  68.92% |     132  15.96% |     702  84.89% |
+-----------------+-----------------+-----------------+-----------------+
| System negative |      30   3.63% |      95  11.49% |     125  15.11% |
+-----------------+-----------------+-----------------+-----------------+
| SUM             |     600  72.55% |     227  27.45% |     827         |
+-----------------+-----------------+-----------------+-----------------+

Baseline: 72.55%
System accuracy: 80.41%

Fold  5 / 10
------------

                  +-----------------+-----------------+-----------------+
                  | Target positive | Target negative | SUM             |
+-----------------+-----------------+-----------------+-----------------+
| System positive |     564  68.20% |     114  13.78% |     678  81.98% |
+-----------------+-----------------+-----------------+-----------------+
| System negative |      50   6.05% |      99  11.97% |     149  18.02% |
+-----------------+-----------------+-----------------+-----------------+
| SUM             |     614  74.24% |     213  25.76% |     827         |
+-----------------+-----------------+-----------------+-----------------+

Baseline: 74.24%
System accuracy: 80.17%

Fold  6 / 10
------------

                  +-----------------+-----------------+-----------------+
                  | Target positive | Target negative | SUM             |
+-----------------+-----------------+-----------------+-----------------+
| System positive |     584  70.62% |     122  14.75% |     706  85.37% |
+-----------------+-----------------+-----------------+-----------------+
| System negative |      28   3.39% |      93  11.25% |     121  14.63% |
+-----------------+-----------------+-----------------+-----------------+
| SUM             |     612  74.00% |     215  26.00% |     827         |
+-----------------+-----------------+-----------------+-----------------+

Baseline: 74.00%
System accuracy: 81.86%

Fold  7 / 10
------------

                  +-----------------+-----------------+-----------------+
                  | Target positive | Target negative | SUM             |
+-----------------+-----------------+-----------------+-----------------+
| System positive |     615  74.37% |     111  13.42% |     726  87.79% |
+-----------------+-----------------+-----------------+-----------------+
| System negative |      28   3.39% |      73   8.83% |     101  12.21% |
+-----------------+-----------------+-----------------+-----------------+
| SUM             |     643  77.75% |     184  22.25% |     827         |
+-----------------+-----------------+-----------------+-----------------+

Baseline: 77.75%
System accuracy: 83.19%

Fold  8 / 10
------------

                  +-----------------+-----------------+-----------------+
                  | Target positive | Target negative | SUM             |
+-----------------+-----------------+-----------------+-----------------+
| System positive |     577  69.77% |     115  13.91% |     692  83.68% |
+-----------------+-----------------+-----------------+-----------------+
| System negative |      39   4.72% |      96  11.61% |     135  16.32% |
+-----------------+-----------------+-----------------+-----------------+
| SUM             |     616  74.49% |     211  25.51% |     827         |
+-----------------+-----------------+-----------------+-----------------+

Baseline: 74.49%
System accuracy: 81.38%

Fold  9 / 10
------------

                  +-----------------+-----------------+-----------------+
                  | Target positive | Target negative | SUM             |
+-----------------+-----------------+-----------------+-----------------+
| System positive |     581  70.25% |     115  13.91% |     696  84.16% |
+-----------------+-----------------+-----------------+-----------------+
| System negative |      49   5.93% |      82   9.92% |     131  15.84% |
+-----------------+-----------------+-----------------+-----------------+
| SUM             |     630  76.18% |     197  23.82% |     827         |
+-----------------+-----------------+-----------------+-----------------+

Baseline: 76.18%
System accuracy: 80.17%

Fold 10 / 10
------------

                  +-----------------+-----------------+-----------------+
                  | Target positive | Target negative | SUM             |
+-----------------+-----------------+-----------------+-----------------+
| System positive |     569  68.80% |     117  14.15% |     686  82.95% |
+-----------------+-----------------+-----------------+-----------------+
| System negative |      45   5.44% |      96  11.61% |     141  17.05% |
+-----------------+-----------------+-----------------+-----------------+
| SUM             |     614  74.24% |     213  25.76% |     827         |
+-----------------+-----------------+-----------------+-----------------+

Baseline: 74.24%
System accuracy: 80.41%

Summary
-------

+---------------------------+---------+----------+
| System better             |      10 |  100.00% |
+---------------------------+---------+----------+
| Baseline better           |       0 |    0.00% |
+---------------------------+---------+----------+
| Average accuracy baseline |             74.62% |
+---------------------------+--------------------+
| Average accuracy system   |             80.90% |
+---------------------------+--------------------+
```
