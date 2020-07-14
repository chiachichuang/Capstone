from turtle import pd

from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
import pickle
# %matplotlib inline

def load_dataset(filename):
	# load the dataset as a pandas DataFrame
	data = read_csv(filename, header=None)
	# retrieve numpy array
	dataset = data.values
	# split into input (X) and output (y) variables
	X = dataset[:, 1:]
	y = dataset[:,0]
	return X, y

def load_dataset_columns(filename,columns):
	# load the dataset as a pandas DataFrame
	data = read_csv(filename, header=None,usecols=columns)
	# retrieve numpy array
	dataset = data.values
	# split into input (X) and output (y) variables
	X = dataset[:, 1:]
	y = dataset[:,0]
	return X, y

def select_features(X_train, y_train):
	# configure to select all features
	fs = SelectKBest(score_func=f_classif, k='all')
	# learn relationship from training data
	fs.fit(X_train, y_train)
	return fs

def plot_eigens(X):
    covariant_matrix = np.cov(X.T)
    #print("==========<<    covariant_matrix    >>===================")
    #print(covariant_matrix)

    eigen_values,eigen_vectors = np.linalg.eig(covariant_matrix)
    # print("==========<<    eigen_vectors    >>===================")
    # print(eigen_vectors)
    # print("==========<<    eigen_values    >>===================")
    # print(eigen_values)

    tot = sum(eigen_values)
    var_exp = [(i/tot) for i in sorted(eigen_values,reverse=True)]
    # print("==========<<    len(var_exp)    >>===================")
    # print(len(var_exp))
    # print("==========<<    var_exp    >>===================")
    # print(var_exp)
    cum_var_exp = np.cumsum(var_exp)
    # print("==========<<    cum_var_exp    >>===================")
    # print(cum_var_exp)

    plt.bar(range(1,36),var_exp,alpha=0.5,align='center',
             label='individual explained variance')
    plt.step(range(1,36),cum_var_exp,where='mid',
        label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.show()

def plot_with_pca(X,df):

    # print("----- X.shape in plot_with_pca ---------------")
    # print(X.shape)
    # print("----- df.shape in plot_with_pca ---------------")
    # print(df.shape)

    num_compnenets=25
    culumn_labels=[]
    for i in range(num_compnenets):
        culumn_labels.append("pc "+str(i+1))
    print("PCA Fit X Site:")
    print(X.shape)
    pca = PCA(n_components=num_compnenets) # 25 components cumulative variance to 0.98
    principalComponents = pca.fit_transform(X) # this is 150x4
    principalDf = pd.DataFrame(data = principalComponents, \
                               columns = culumn_labels)

    filename = 'mnist_pca_capstone.pkl'
    pickle.dump(pca, open(filename, 'wb'))

    # print("-------------   principalDf.shape   -------------")
    # print(principalDf.shape)
    # print("-------------   pca.explained_variance_ratio_   -------------")
    # print(pca.explained_variance_ratio_)
    # print("-------------   pca.components_   -------------")
    # print(pca.components_)

    # PC1 = (0.36*SL)-(0.08*SW)+(0.85*PL)+(0.35*PW)
    # PC2 = (0.65*SL)+(0.72*SW)-(0.17*PL)-(0.07*PW)
    # print("-------------   principalDf.head(num_compnenets)   -------------")
    # # print(principalDf.head(num_compnenets))
    # print(principalDf.head(2))

    finalDf = pd.concat([principalDf, df[['num_cat']]], axis = 1)
    print("-------------  After set finalDF   -------------")

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

def classifier_with_pca(finalDf_train):
    #def classifier_with_pca(finalDf_train, finalDf_test):
    # PCA with 25 Principal Components = 98% of data variance
    X_pca_train = finalDf_train.loc[:,'pc 1':'pc 25']
    y_pca_train = finalDf_train['num_cat']

    # Decision Tree Classifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import metrics
    model = DecisionTreeClassifier()
    model.fit(X_pca_train, y_pca_train)

    print("====<< PCA Model >>==============")
    print(model)

    X_test, y_test = load_dataset('mnist_test.csv')
    X_test, y_test = np.array(X_test), np.array(y_test)
    print(X_test.shape)
    print(y_test.shape)

    filename = 'mnist_pca_capstone.pkl'
    loaded_pca_model = pickle.load(open(filename, 'rb'))
    principalComponents_test = loaded_pca_model.transform(X_test)
    column_input = []
    for i in range(25):
        column_input.append("pc " + str(i + 1))
    principalDf_test = pd.DataFrame(data=principalComponents_test, \
                                    columns=column_input)
    principalDf_test["num_cat"] = y_test

    # make predictions
    # X_pca_test = finalDf_test.loc[:, 'pc 1':'pc 25']
    # y_pca_test = finalDf_test['num_cat']
    expected = y_test
    predicted = model.predict(principalDf_test)
    # summarize the fit of the model
    # print("------------- metrics.classification_report(expected, predicted)  -------------")
    # print(metrics.classification_report(expected, predicted))

    cm = metrics.confusion_matrix(expected, predicted)
    # print("-------------  metrics.confusion_matrix(expected, predicted) -------------")
    # print(metrics.confusion_matrix(expected, predicted))
    # idx = 1
    # for item in cm:
    #     print("((((((( " + str(idx) + "   ))))))))))))")
    #     print(item)
    #     idx += 1

def main():
    X_train, y_train = load_dataset('mnist_train.csv')
    X_test, y_test = load_dataset('mnist_test.csv')

    # feature selection
    fs = select_features(X_train, y_train)
    selected_features = []
    for i in range(len(fs.scores_)):
        if fs.scores_[i] > 3000:
            selected_features.append(i + 1)

    selected_feature_size = len(selected_features)
    print("selected_features : size (" + str(selected_feature_size) + ")")
    pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
    pyplot.show()
    pyplot.close()

    # feature selection
    fs_test = select_features(X_test, y_test)
    selected_features_test, track = [], []
    for i in range(len(fs_test.scores_)):
        if fs_test.scores_[i] > 0.000000001:
            track.append([fs_test.scores_[i], i+1])
    track.sort(reverse=True)
    for i in range(selected_feature_size):
        selected_features_test.append(track[i][1])
    print("selected_features_test : size (" + str(len(selected_features_test)) + ")")
    pyplot.bar([i for i in range(len(fs_test.scores_))], fs_test.scores_)
    pyplot.show()
    pyplot.close()

    ####
    # load the dataset with selected feature
    X_imp3, y3 = load_dataset_columns('mnist_train.csv', columns = selected_features3)
    print(X_imp3.shape, y3.shape)

    #X_test, y_test = load_dataset_columns('mnist_test.csv', columns = selected_features_test)
    X_test, y_test = load_dataset_columns('mnist_test.csv', columns=selected_features3)
    print(X_test.shape, y_test.shape)

    print("------------------- Call def plot_eigens  & plot_with_pca (X_imp3) ---------------------")
    ds_X = np.array(X_imp3)
    ds_y = np.array(y3)
    plot_eigens(ds_X)
    df = pd.DataFrame(ds_X)
    df["num_cat"] = ds_y
    finalDf = plot_with_pca(ds_X, df)

    print("------------------- Call def plot_eigens vs plot_with_pca (X_test) ---------------------")
    ds_X_test = np.array(X_test)
    ds_y_test = np.array(y_test)
    plot_eigens(ds_X_test)
    df_test = pd.DataFrame(ds_X_test)
    df_test["num_cat"] = ds_y_test
    finalDf_test = plot_with_pca(ds_X_test, df_test)

    #pc_test = loaded_pca_model.transform(Xtest)
    classifier_with_pca(finalDf, finalDf_test)



main()