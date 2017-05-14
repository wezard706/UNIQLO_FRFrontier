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
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import grid_search
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score, KFold
from preprocess import *
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

    '''
    # label binarizer
    # encode label to one hot representation
    label = data['category_id'].as_matrix().reshape(data.shape[0], 1)
    label_onehot = pd.DataFrame(LabelBinarizer().fit_transform(label))
    data = pd.concat([data['file_name'], label_onehot], axis=1)
    '''

    X = []
    y = []
    for row in data.iterrows():
        # f, l = row[1]['file_name'], row[1].ix[1:].as_matrix().astype('int64') label binarizer
        f, l = row[1]['file_name'], row[1]['category_id']
        try:
            # im = Image.open(os.path.join(path_to_train_images, f))
            im = cv2.imread(os.path.join(path_to_train_images, f))
            if IS_CONTOUR:
                im_contour = extract_clothes_color(im)
                X.append(im_contour)
                y.append(l)
            else:
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
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=inverse_onehot(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    # grid search
    parameters = {
        'n_estimators': [100],
        'max_features': [64],
        'max_depth': [None]
    }
    # cv = KFold(n=len(X_train), n_folds=5, shuffle=True) # label binarizer
    # clf = grid_search.GridSearchCV(OneVsRestClassifier(RandomForestClassifier()), parameters, cv=cv, scoring='precision', n_jobs=-1) # label binarizer
    # clf = grid_search.GridSearchCV(RandomForestClassifier(), parameters, cv=5, scoring='precision', n_jobs=-1)
    clf = grid_search.GridSearchCV(RandomForestClassifier(), parameters, cv=5, n_jobs=-1)
    clf.fit(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    y_test_pred = clf.predict(X_test)


    ''' # label binarizer
    y_test_pred = clf.predict_proba(X_test)
    y_test_pred = y_test_pred.argmax(axis=1)
    y_test = y_test.argmax(axis=1)
    '''

    conf_mat = confusion_matrix(y_test, y_test_pred, labels=range(23))
    print conf_mat
    print ("prediction from confusion matrix: {}").format(1.0 * np.diag(conf_mat).sum() / conf_mat.sum())
    plt.matshow(conf_mat)
    plt.colorbar()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.savefig("confusion_matrix.jpg")

    # print result of grid search
    gs_max_features, gs_n_estimators, gs_max_depth = clf.best_params_.values()
    print "best parameter: {}".format(clf.best_params_)
    print "best score: {}".format(clf.best_score_)
    print "average score:\n"
    for params, mean_score, all_scores in clf.grid_scores_:
        print "{:.3f} (+/- {:.3f}) for {}".format(mean_score, all_scores.std() / 2, params)
    print "test score: {}".format(test_score)

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
