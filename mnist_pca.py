from turtle import pd

#from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
import pickle

def load_dataset(filename):
    # load the dataset as a pandas DataFrame
    data = read_csv(filename, header=None)
    # retrieve numpy array
    dataset = data.values
    # split into input (X) and output (y) variables
    # X = dataset[:, 1:]
    # y = dataset[:,0]
    return dataset[:, 1:], dataset[:,0]

def load_dataset_columns(filename, columns):
    # load the dataset as a pandas DataFrame
    selected_col = [0]
    for col_idx in columns:
        selected_col.append(col_idx)
    data = read_csv(filename, header=None, usecols=selected_col)
    # retrieve numpy array
    dataset = data.values
    # split into input (X) and output (y) variables
    # X = dataset[:, 1:]
    # y = dataset[:,0]
    return dataset[:, 1:], dataset[:,0]

def select_features(X_train, y_train):
	# configure to select all features
	fs = SelectKBest(score_func=f_classif, k='all')
	# learn relationship from training data
	fs.fit(X_train, y_train)
	return fs

def plot_eigens(X):
    covariant_matrix = np.cov(X.T)
    eigen_values,eigen_vectors = np.linalg.eig(covariant_matrix)
    # print("==========<<    eigen_vectors    >>===================")
    # print(eigen_vectors)
    # print("==========<<    eigen_values    >>===================")
    # print(eigen_values)

    tot = sum(eigen_values)
    var_exp = [(i/tot) for i in sorted(eigen_values,reverse=True)]
    print("==========<<    len(var_exp)    >>===================")
    print(len(var_exp))
    print("==========<<    var_exp    >>===================")
    print(var_exp)
    cum_var_exp = np.cumsum(var_exp)
    print("==========<<    cum_var_exp    >>===================")
    print(cum_var_exp)
    #37 for 36 selected feature, feature val > 3000
    #198 for 197 selected feature, feature val > 1500
    # 80 for 79 selected feature, feature val > 2500
    plt.bar(range(1, 198), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(1, 198), cum_var_exp, where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.show()

def plot_with_pca(X,df, saveModel = False):
    num_compnenets=25
    culumn_labels = []
    for i in range(num_compnenets):
        culumn_labels.append("pc "+str(i+1))
    print("PCA X Shape:")
    print(X.shape)
    filename = 'mnist_pca_capstone.pkl'
    if saveModel:
        # 25 components cumulative variance roughly equals 0.98 for 36 selected feauters (>3000 val)
        # 60 components cumulative variance roughly equals 0.98 for 197 selected feauters (>1500 val)
        # but 25 seems result is better than 60 or 35
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
        # print(principalComponents.shape)

        # column_input = []
        # for i in range(25):
        #     column_input.append("pc " + str(i + 1))
        # principalDf_test = pd.DataFrame(data=principalComponents_test, \
        #                                 columns=column_input)
        # principalDf_test["num_cat"] = y_test

    principalDf = pd.DataFrame(data=principalComponents, \
                               columns=culumn_labels)

    finalDf = pd.concat([principalDf, df[['num_cat']]], axis = 1)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('pc 1', fontsize = 15)
    ax.set_ylabel('pc 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = [i for i in range(10)]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['num_cat'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'pc 1'],
                   finalDf.loc[indicesToKeep, 'pc 2'],
                   c = color)
    ax.legend(targets)
    ax.grid()
    plt.show()

    return finalDf

def classifier_with_pca(finalDf, finalDf_test):
    # X_pca = finalDf.loc[:,'pc 1':'pc 25']
    # y_pca = finalDf['num_cat']
    # from sklearn.tree import DecisionTreeClassifier
    # from sklearn import metrics
    model = DecisionTreeClassifier()
    model.fit(finalDf.loc[:, 'pc 1':'pc 25'],finalDf['num_cat'])
    print("----------------- model ---------------------- ")
    print(model)
    # make predictions
    expected = finalDf_test['num_cat']
    predicted = model.predict(finalDf_test.loc[:,'pc 1':'pc 25'])
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))


def main():
    X_train, y_train = load_dataset('mnist_train.csv')
    X_test, y_test = load_dataset('mnist_test.csv')

    # feature selection
    fs = select_features(X_train, y_train)
    selected_features = []
    for i in range(len(fs.scores_)):
        #if fs.scores_[i] > 3000:
        #if fs.scores_[i] > 1500:
        #if fs.scores_[i] > 2500:
        if fs.scores_[i] > 1500:
            selected_features.append(i + 1)

    selected_feature_size = len(selected_features)
    print("selected_features : size (" + str(selected_feature_size) + ")")
    pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
    pyplot.show()
    pyplot.close()

    ####
    # load the data-set with selected feature
    X_imp3, y3 = load_dataset_columns('mnist_train.csv', columns = selected_features)
    X_test, y_test = load_dataset_columns('mnist_test.csv', columns=selected_features)

    ds_X = np.array(X_imp3)
    ds_y = np.array(y3)
    plot_eigens(ds_X)
    df = pd.DataFrame(ds_X)
    df["num_cat"] = ds_y
    finalDf = plot_with_pca(ds_X, df, True)
    finalDf["num_cat"] = ds_y

    ds_X_test = np.array(X_test)
    ds_y_test = np.array(y_test)
    plot_eigens(ds_X_test)
    df_test = pd.DataFrame(ds_X_test)
    df_test["num_cat"] = ds_y_test
    finalDf_test = plot_with_pca(ds_X_test, df_test)
    finalDf_test["num_cat"] = ds_y_test

    classifier_with_pca(finalDf, finalDf_test)


main()