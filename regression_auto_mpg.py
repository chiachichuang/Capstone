
# Feature selection and application to ML Regression

# Using ANOVA f-test: It is a ratio of variances between
# columns and captures which columns are responsible for the
# most dataset variance.

# The goal is to keep column which contribute to information
# gain or data variance and drop the ones which do not.

#import pandas as pd
from numpy import int64
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from matplotlib import pyplot

"""
data source : http://archive.ics.uci.edu/ml/datasets/Auto+MPG

The data concerns city-cycle fuel consumption in miles per gallon, 
to be predicted in terms of 3 multivalued discrete and 5 continuous attributes." 
(Quinlan, 1993)

1. mpg: continuous
2. cylinders: multi-valued discrete
3. displacement: continuous
4. horsepower: continuous (most int, but ? sometimes)
5. weight: continuous
6. acceleration: continuous
7. model year: multi-valued discrete
8. origin: multi-valued discrete
9. car name: string (unique for each instance)
-----------------------------------------------------------------------
1. Number of Instances: 398
2. Number of Attributes: 8 plus target mpg (city-cycle fuel consumption)

"""
def get_corr(filename):
    # load the dataset as a pandas DataFrame
    df = read_csv(filename, usecols=[0,1,2,3,4,5,6,7], names=['Y', 'f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6'])
    #print(df.dtypes)
    #print(df.shape)
    df = df.loc[lambda rw: rw['f2'] != "?"]
    df.f2 = df.f2.astype(int64)

    print("=======================  Data General Information ================")
    print(df.describe())

    aaa = df.corr(method='kendall')
    print("\n\n=======>>> Corr matrix ...\n",aaa)

# load the dataset
def load_dataset(filename):
    # load the dataset as a pandas DataFrame
    data = read_csv(filename, header=None)
    data = data.loc[lambda df: df[3] != "?"]
    # retrieve numpy array
    dataset = data.values
    # split into input (X) and output (y) variables
    X = dataset[0:, 1:8]
    y = dataset[0:, 0:1]

    return X, y

# load the dataset with selected columns only
def load_dataset_columns(filename,columns):
    # load the dataset as a pandas DataFrame
    data = read_csv(filename, header=None,usecols=columns)
    # retrieve numpy array
    dataset = data.values
    # split into input (X) and output (y) variables
    X = dataset[0:, 1:]
    y = dataset[0:, 0:1]
    return X, y

# feature selection
# share more doc with class
def select_features(X_train, y_train):
    # configure to select all features
    fs = SelectKBest(score_func=f_classif, k='all')
    # learn relationship from training data
    fs.fit(X_train, y_train)
    return fs

# Run ML classification with logistic regression
# return confusion matrix
def regress(X_train_imp,y_train,X_test,y_test):
        import math
        from sklearn.metrics import mean_squared_error
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.linear_model import LassoLars, BayesianRidge
        from sklearn.linear_model import ElasticNet, Lars
        # Next line is "dictionary" data structure from class 1
        runs = []
        # https://scikit-learn.org/stable/modules/linear_model.html
        # 1.1.1 LinearRegression
        # 1.1.2 Ridge
        # 1.1.3 Lasso
        # 1.1.8 LassoLars
        # 1.1.10 BayesianRidge

        # 1.1.5 ElasticNet
        # 1.1.7 Lars

        d_models = {"Linear_Regression": LinearRegression(),
                     "Ridge": Ridge(alpha=0.5),
                     "Lasso": Lasso(alpha=0.1),
                     "LassoLars": LassoLars(alpha=0.1),
                     "BayesianRidge": BayesianRidge(),
                     "ElasticNet": ElasticNet(alpha=0.5, l1_ratio=0.7),
                     "Lars": Lars(n_nonzero_coefs=3)}
        models_list = d_models.keys()
        print("---- models lists ---")
        print(models_list,"\n")

        for regression_model in models_list:
            regressor = d_models[regression_model]
            regressor.fit(X_train_imp,y_train)
            y_predict = regressor.predict(X_test)
            regression_model_mse = mean_squared_error(y_predict, y_test)
            sqrt_regression_model_mse = math.sqrt(regression_model_mse)
            print(regression_model,"\nRMSE:",sqrt_regression_model_mse)
            runs.append(sqrt_regression_model_mse)
            print(regressor.coef_)
            print(regressor.intercept_)
            print("=======")
        return runs
    
def main():
        input_file_name = 'auto-mpg.csv'
        # load the dataset
        X, y = load_dataset(input_file_name)
        print(X.shape,y.shape)

        # split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

        # feature selection
        fs = select_features(X_train, y_train)

        print("==================  Features Information =====================")
        # what are scores for the features
        for i in range(len(fs.scores_)):
                print('Feature %d: %f' % (i, fs.scores_[i]))
        # plot the scores
        pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
        pyplot.show()
        pyplot.close()

        print("=============  After reduced from 7 to 4 features : Features 0, 1, 3 selected ===========")
        get_columns = [0, 1, 2, 4]
        # X_imp are important columns
        X, y = load_dataset_columns(input_file_name, columns=get_columns)
        X_train_imp, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

        print(" <<< X, y >>>", X.shape, y.shape)
        print(" <<< X_train_imp, y_train >>>", X_train_imp.shape, y_train.shape)
        print(" <<< X_test, y_test >>>", X_test.shape, y_test.shape)

        runs = regress(X_train_imp,y_train,X_test,y_test)
        print(runs)

        get_corr(input_file_name)

main()
    
