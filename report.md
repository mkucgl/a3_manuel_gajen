Programming Assignment 3
========================

Handed in by Manuel Kunz and Gajendira Sivajothi

Introduction
------------

The code that we produced to solve the programming assignment can be found in the file "assignment3.py". It can be run like this...

Documentation of Development
----------------------------

### Exercise 1

1. IMD_Resolver
2. Uncached entity resolution function
3. Created functions for storing loading entity cache
4. Cached entity resolution function
5. Populated entity cache
6. Decision whether item is positive item
7. Identification of entity name in snippet
8. Split corpus: development/training/test data set (see [Data Sets](#data-sets) below).

### Exercise 2

1. Match counts as first features

### Exercise 3

1. Including and first try of LogisticRegression: fit, getting coefficients and intercept

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

Appendix 1: File Overview
-------------------------

Code:

- assignment3.py: The main Python script containing the program that is our solution of
  the third programming assignment doing classification, testing and validation. It also
  contains some code for preparing the features (described below).
- IMD_resolver.py: Resolver for entity ids derived from the example provided in
  ResolvingIDs.docx

Entity cache:

- entity_cache.json: Contains all the data downloaded during the first resolution of all
  ids so that they don't have to be resolved again

Corpora / data sets:

- 20130403-institutions.json: The completed institution corpus as downloaded from Google.
- 20130403-place_of_birth.json: The complete place of birth corpus as downloaded from
  Google

General info:

- ML-Tips.ipynb and ML-Tips.py: Tips from the OLAT material folder
- ResolvingIDs.docx: Introduction/example/info of/about id resolution downloaded from
  material folder
- exercise_text.txt: The exercise text as downloaded from OLAT

Report:

- report.md /report.html: Report in different formats
