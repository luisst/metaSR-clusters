import os
import numpy as np
import pickle
import sys
from pathlib import Path
import constants
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
import re





test_feat_dir = constants.CENTROID_FEAT_AOLME

noise_type = 'onlySpeech'
dataset_type = 'pcaPRE_AOLME_groups_NORM'


with open(f'dVectors_{dataset_type}_noise{noise_type}.pickle', "rb") as file:
    X_data_and_labels = pickle.load(file)
# Mixed_X_data, Mixed_X_paths, Mixed_y_labels = X_data_and_labels

X_data, y_labels = X_data_and_labels
print(f'Original shape: {X_data.shape}')

# Define classifiers and feature selection methods with their parameters
methods = [
    (SelectFromModel(LinearSVC(dual=False, penalty="l2")), RandomForestClassifier()),
    (SelectFromModel(LinearSVC(dual=False, penalty="l2")), SVC()),
    (SelectFromModel(LogisticRegression(penalty="l1", solver='liblinear', max_iter=5000)), RandomForestClassifier()),
    (SelectFromModel(LinearSVC(dual=False, penalty="l2")), DecisionTreeClassifier()),
    (SelectFromModel(LinearSVC(dual=False, penalty="l2")), KNeighborsClassifier()),
    (SelectFromModel(LinearSVC(dual=False, penalty="l2")), GradientBoostingClassifier()),
    (SelectFromModel(LinearSVC(dual=False, penalty="l2")), AdaBoostClassifier()),
    (SelectFromModel(LinearSVC(dual=False, penalty="l2")), XGBClassifier()),
]
# min_columns = float('inf')
# best_method = None

# for feature_selection, classification in methods:
#     clf = Pipeline([
#         ('feature_selection', feature_selection),
#         ('classification', classification)
#     ])
#     clf.fit(X_data, y_labels)
#     X_new = clf.named_steps['feature_selection'].transform(X_data)
#     new_shape = X_new.shape[1]
#     print(f'New shape: {new_shape} with method: {feature_selection} and classifier: {classification}')
    
#     if new_shape < min_columns:
#         min_columns = new_shape
#         best_method = (feature_selection, classification)

# print(f'Best method: {best_method} with shape: {min_columns}')

my_new_X = feature_selection_methods(methods[0], X_data, y_labels)
print(f'New shape: {my_new_X.shape}')