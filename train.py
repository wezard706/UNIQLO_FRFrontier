# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:50:58 2017

@author: n.aoi
"""

import os
import cv2
import pandas as pd
import numpy as np
import cPickle as pickle
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import grid_search
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score, KFold
from preprocess import *
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
# from package.models import MyModel # you may import your own model in the package
# from package import preprocess_methods # you may import your own preprocess method in the package

PATH_TO_TRAIN_IMAGES = os.path.join('train')
PATH_TO_TRAIN_DATA = os.path.join('train_master_short.tsv')
PATH_TO_MODEL = os.path.join('models', 'random_forest')
PATH_TO_TRAIN_DATA_NP = os.path.join('train')

IS_CONTOUR = False

def load_train_data(path_to_train_images, path_to_train_data):
    print('loading train data from csv...')
    data = pd.read_csv(path_to_train_data, sep=',')

    X = []
    y = []
    for row in data.iterrows():
        f, l = row[1]['file_name'], row[1]['category_id']
        try:
            im = cv2.imread(os.path.join(path_to_train_images, f))
            X.append(np.array(im).flatten())
            y.append(l)
        except Exception as e:
            print(str(e))

    X = np.array(X)
    y = np.array(y)
    print('done.')
    return X, y

def inverse_onehot(vec):
    return np.where(vec)[1]

def train_model(X, y):
    print('training the model ...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    # grid search
    '''
    parameters = {
        'n_estimators': [100, 200, 300],
        'max_features': [64, 128, 256],
        'max_depth': [5, 10, None]
    }
    '''

    parameters = {
        'n_estimators': [100],
        'max_features': [64],
        'max_depth': [None]
    }

    '''
    parameters = {
        'C': [1],
        'gamma': [0.001]
    }
    '''
    clf = GridSearchCV(RandomForestClassifier(class_weight='balanced'), parameters, cv=5, n_jobs=-1, scoring='f1_weighted')
    clf.fit(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    y_test_pred = clf.predict(X_test)


    # visualization
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    conf_mat = confusion_matrix(y_test, y_test_pred, labels=range(23))
    conf_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    print conf_norm
    print ("prediction from confusion matrix: {}").format(1.0 * np.diag(conf_norm).sum() / conf_norm.sum())
    ax1.matshow(conf_norm)
    ax1.set_xlabel("Predicted label")
    ax1.set_ylabel("True label")

    ax2.hist(y_train, bins=24, range=(0, 23))
    plt.savefig("static_info.jpg")

    # print result of grid search
    gs_max_features, gs_n_estimators, gs_max_depth = clf.best_params_.values()
    print "best parameter: {}".format(clf.best_params_)
    print "best score: {}".format(clf.best_score_)
    print "average score:\n"
    for params, mean_score, all_scores in clf.grid_scores_:
        print "{:.3f} (+/- {:.3f}) for {}".format(mean_score, all_scores.std() / 2, params)
    print "test score: {}".format(test_score)

    # the number of image for each label
    for i in range(24):
        print('label {}: {}').format(i, y_test[y_test == i].shape[0])

    return clf

def save_model(model, name):
    print('saving the model ...')
    with open(name+'.pkl', mode='wb') as f:
        pickle.dump(model, f)
    print('done.')

def save_data(data, name):
    print('saving the data ...')
    with open(name + '.pkl', mode='wb') as f:
        pickle.dump(data, f)
    print('done.')

if __name__ == '__main__':
    ## load the data for training
    X, y = load_train_data(PATH_TO_TRAIN_IMAGES, PATH_TO_TRAIN_DATA)

    ## make color histogram from train images
    X_hist = make_hist(X)

    ## instanciate and train the model
    model = train_model(X_hist, y)

    ## save the trained model
    save_model(model, PATH_TO_MODEL)
