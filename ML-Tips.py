
# coding: utf-8

# ## Useful scikit-learn tooboxes:
# 
# from sklearn.model_selection:
# 
# * train_test_split, KFold, StratifiedKFold ---> data split
# 
# * cross_val_score ---> evaluation of a given score using CV (pay attention to scoring and cv parameters)
# 
# * GridSearchCV ---> hyperparameters tuning (this is not required for this task but feel free to explore)
# 
# 
# from sklearn.feature_extraction:
# 
# * CountVectorizer --> extract bag-of-word features from text (Text is described by word occurrences while completely ignoring the relative position information of the words in the text). Converts a sequence(i.e list, numpy..) of strings to a SciPy representation of the counts used by scikit-learn estimators. Some parameters, e.g. tokenization ,stop words can be overwritten.
# 
# * DictVectorizer --> convert feature arrays represented as lists of standard Python dict objects to the NumPy/SciPy representation used by scikit-learn estimators
# 
# 
# from sklearn.pipeline:
# 
# * FeatureUnion --> combine different features together. Here there is an interesting example how to present data before feeding it into a pipeline: http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html#
# 
# 
# from sklearn.metrics: 
# 
# * classification_report, confusion_matrix --> could be used for performance analysis on development set
# 
# 
# ## Inspiration links:
# 
# inspiration for mentions detecion:
# 
# http://stackoverflow.com/questions/425604/best-way-to-determine-if-a-sequence-is-in-another-sequence-in-python --> sequence in sequence
# 
# difflib.SequenceMatcher --> sequence matching
# 
# 
# inspiration for the use of syntactic information:
# 
# https://nicschrading.com/project/Intro-to-NLP-with-spaCy/
# 
# http://stackoverflow.com/questions/33289820/noun-phrases-with-spacy
# 
# https://demos.explosion.ai/displacy/
# 
# https://demos.explosion.ai/displacy-ent/
# 
# http://stackoverflow.com/questions/32835291/how-to-find-the-shortest-dependency-path-between-two-words-in-python
# 
# 
# Notebooks and code for the book "Introduction to Machine Learning with Python":
# 
# https://github.com/amueller/introduction_to_ml_with_python --> check out visualization of feature coefficients, Ch 7 
# 
# 
