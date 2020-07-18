"""
data source :
https://www.kaggle.com/mlg-ulb/creditcardfraud?select=creditcard.csv

meta data:
31 columns
Time: Number of seconds elapsed between this transaction and the first transaction in the dataset
V1 - V28: may be result of a PCA Dimensionality reduction to protect user identities and sensitive features(v1-v28)
Amount: transaction amount
Class: 1 for fraudulent transactions, 0 otherwise
"""
import pandas as pd
from numpy import int64
from pandas import read_csv
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import matplotlib.pyplot as plt
import pickle

#from turtle import pd

#from matplotlib import pyplot


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

import numpy as np



def load_dataset(filename):
    # load the dataset as a pandas DataFrame
    data = read_csv(filename, header=0)
    data.Time = data.Time.astype(int64)
    print(data.describe())
    # retrieve numpy array
    dataset = data.values
    return dataset[:, :30], dataset[:,30]

def load_dataset_columns(filename, columns):
    # load the dataset as a pandas DataFrame
    data = read_csv(filename, header=0, usecols=columns)
    print("====================  After Feature selections  ====================")
    print(data.describe())
    # retrieve numpy array
    dataset = data.values
    upb = len(columns)
    return dataset[:, :upb-1], dataset[:, upb-1]

def select_features(X_train, y_train):
	# configure to select all features
	fs = SelectKBest(score_func=f_classif, k='all')
	# learn relationship from training data
	fs.fit(X_train, y_train)
	return fs

def plot_eigens(X):
    covariant_matrix = np.cov(X.T)
    eigen_values,eigen_vectors = np.linalg.eig(covariant_matrix)
    print("==========<<    eigen_vectors    >>===================")
    print(eigen_vectors)
    print("==========<<    eigen_values    >>===================")
    print(eigen_values)

    tot = sum(eigen_values)
    var_exp = [(i/tot) for i in sorted(eigen_values,reverse=True)]
    print("==========<<    len(var_exp)    >>===================")
    print(len(var_exp))
    print("==========<<    var_exp    >>===================")
    print(var_exp)
    cum_var_exp = np.cumsum(var_exp)
    print("==========<<    cum_var_exp    >>===================")
    print(cum_var_exp)
    plt.bar(range(1, 8), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(1, 8), cum_var_exp, where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.show()

def plot_with_pca(X,df, saveModel = False):
    num_compnenets=7
    culumn_labels = []
    for i in range(num_compnenets):
        culumn_labels.append("principal component " + str(i+1))
    print("PCA X Shape:")
    print(X.shape)
    filename = 'fraud_trans_detect.pkl'
    if saveModel:
        pca = PCA(n_components=num_compnenets)
        principalComponents = pca.fit_transform(X)
        pickle.dump(pca, open(filename, 'wb'))
        print("-------------   pca.explained_variance_ratio_   -------------")
        print(pca.explained_variance_ratio_)
        print("-------------   pca.components_   -------------")
        print(pca.components_)
    else:
        loaded_pca_model = pickle.load(open(filename, 'rb'))
        principalComponents = loaded_pca_model.transform(X)

    principalDf = pd.DataFrame(data=principalComponents, \
                               columns=culumn_labels)

    finalDf = pd.concat([principalDf, df[['Class']]], axis = 1)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('principal component 1', fontsize = 15)
    ax.set_ylabel('principal component 2', fontsize = 15)
    ax.set_title('2 components PCA', fontsize = 20)
    targets = [0,1]
    colors = ['green', 'red']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['Class'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                   finalDf.loc[indicesToKeep, 'principal component 2'],
                   c = color)
    ax.legend(targets)
    ax.grid()
    plt.show()

    return finalDf

def classifier_with_pca(finalDf, finalDf_test):
    model = DecisionTreeClassifier()
    model.fit(finalDf.loc[:, 'principal component 1':'principal component 7'],finalDf['Class'])
    print("----------------- model ---------------------- ")
    print(model)
    # make predictions
    expected = finalDf_test['Class']
    predicted = model.predict(finalDf_test.loc[:,'principal component 1':'principal component 7'])
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))

def show_features_bar(fs):
    # scores for the features in bar chart
    for i in range(len(fs.scores_)):
        print('Feature %d: %f' % (i, fs.scores_[i]))
    # plot the scores
    pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
    pyplot.show()
    pyplot.close()

def get_selected_features(fs, lowest_accepted_score):
    selected_features = []
    for i in range(len(fs.scores_)):
        if fs.scores_[i] > lowest_accepted_score:
            selected_features.append(i)
    selected_feature_size = len(selected_features)
    print("selected_features : size (" + str(selected_feature_size) + ")")
    return selected_features

def main():
    # load the dataset
    X, y = load_dataset('creditcard.csv')
    print("dataset shape (X, y) :")
    print(X.shape, y.shape, "\n")

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    print("Train dataset shape (X_train, y_train) :: ")
    print(X_train.shape, y_train.shape, "\n")
    print("Test dataset shape (X_test, y_test):: ")
    print(X_test.shape, y_test.shape, "\n")


    # feature selection
    fs = select_features(X_train, y_train)
    show_features_bar(fs)
    # what are scores for the features
    # for i in range(len(fs.scores_)):
    #     print('Feature %d: %f' % (i, fs.scores_[i]))
    # # plot the scores
    # pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
    # pyplot.show()
    # pyplot.close()

    lowest_accepted_score = 8800
    selected_features = get_selected_features(fs, lowest_accepted_score)
    # selected_features = []
    # for i in range(len(fs.scores_)):
    #     if fs.scores_[i] > 8800:
    #         selected_features.append(i)
    # selected_feature_size = len(selected_features)
    # print("selected_features : size (" + str(selected_feature_size) + ")")

    # load the data-set with selected feature
    selected_features.append(30) # put back the Class column
    X_selected, y_selected = load_dataset_columns('creditcard.csv', columns = selected_features)
    print(">>Features selected<< dataset shape (X_test, y_test):: ")
    print(X_selected.shape, y_selected.shape, "\n")

    # split into train and test sets
    X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(X_selected, y_selected, test_size=0.25, random_state=1)
    print(">>Features selected<< Train dataset shape (X_s_train, y_s_train) :: ")
    print(X_s_train.shape, y_s_train.shape, "\n")
    print(">>Features selected<< Test dataset shape (X_s_test, y_s_test):: ")
    print(X_s_test.shape, y_s_test.shape, "\n")

    ds_X = np.array(X_s_train)
    ds_y = np.array(y_s_train)
    plot_eigens(ds_X)
    df = pd.DataFrame(ds_X)
    df["Class"] = ds_y
    finalDf = plot_with_pca(ds_X, df, True)
    finalDf["Class"] = ds_y

    ds_X_test = np.array(X_s_test)
    ds_y_test = np.array(y_s_test)
    plot_eigens(ds_X_test)
    df_test = pd.DataFrame(ds_X_test)
    df_test["Class"] = ds_y_test
    finalDf_test = plot_with_pca(ds_X_test, df_test)
    finalDf_test["Class"] = ds_y_test

    classifier_with_pca(finalDf, finalDf_test)

main()