from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

"""
1. Number of Instances: 150

2. Number of Attributes: 4 plus class

3. For Each Attribute: (all numeric-valued)
   1. sepal length in cm
   2. sepal width in cm
   3. petal length in cm
   4. petal width in cm
   5. class (target):
      1 -- Iris Setosa
      2 -- Iris Versicolour
      3 -- Iris Virginica

   Class Value  Number of instances
   1            50  Iris Setosa
   2            50  Iris Versicolour
   3            50  Iris Virginica
"""

Iris_types = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

# load the dataset
def load_dataset(filename):
    global Iris_types
    data = read_csv(filename, header = None)
    dataset = data.values
    X = dataset[:, :-1]
    y = dataset[:,-1]
    # for idx,type in enumerate(Iris_types):
    #     y[y==type]=idx
    return X, y

# feature selection
def select_features(X_train, y_train):
	# configure to select all features
	fs = SelectKBest(score_func=f_classif, k='all')
	# learn relationship from training data
	fs.fit(X_train, y_train)
	return fs

# load the dataset with selected columns only
def load_dataset_columns(filename, columns):
    global Iris_types
    data = read_csv(filename, header = None, usecols = columns)
    # retrieve numpy array
    dataset = data.values
    X = dataset[:, :-1]
    y = dataset[:,-1]
    # for idx,type in enumerate(Iris_types):
    #     y[y==type]=idx
    return X, y


# Run ML classification with logistic regression
# return confusion matrix
def logreg(X_train, y_train, X_test, y_test, my_solver, extra=None):
    # Logistic Regression
    # fit a logistic regression model to the data
    if extra == None:
        model = LogisticRegression(max_iter=5000, solver=my_solver)
    else:
        model = LogisticRegression(max_iter=5000, solver=my_solver,class_weight=extra)
    model.fit(X_train, y_train)
    print(model)
    # make predictions
    predicted = model.predict(X_test)
    # summarize the fit of the model
    print("============", my_solver, "============")
    print(metrics.classification_report(y_test, predicted))
    cm = metrics.confusion_matrix(y_test, predicted)
    print(cm)
    print("============")
    return cm


def main():

    # load the dataset
    X, y = load_dataset('Iris_dataset.csv')
    print(X.shape, y.shape)

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    # feature selection
    fs = select_features(X_train, y_train)

    # what are scores for the features
    for i in range(len(fs.scores_)):
        print('Feature %d: %f' % (i, fs.scores_[i]))
    # plot the scores
    pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
    pyplot.show()
    pyplot.close()

    runs = []
    solvers = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
    for solver in solvers:
        cm = logreg(X_train, y_train, X_test, y_test, solver)
        result = cm.ravel()  # returns flattened array
        trues = result[0] + result[4] + result[8]
        runs.append(trues)

    # load the dataset with imported X columns (2,3) and column 4 (target)
    get_columns = [2,3,4]
    X_imp, y = load_dataset_columns('Iris_dataset.csv', columns=get_columns)
    print(X_imp.shape, y.shape)

    # split into train and test sets
    X_imp_train, X_imp_test, y_train, y_test = train_test_split(X_imp, y, test_size=0.25, random_state=1)

    for solver in solvers:
        cm = logreg(X_imp_train, y_train, X_imp_test, y_test, solver)
        result = cm.ravel()  # returns flattened array
        trues = result[0] + result[4] + result[8]
        runs.append(trues)

    #print("... Summary of model correct predictions ....\n", runs)

    X_imp_train, X_imp_test, y_train, y_test = train_test_split(X_imp, y, test_size=0.25, random_state=1)
    runs2 = []
    for solver in solvers:
        cm = logreg(X_imp_train, y_train, X_imp_test, y_test, solver, "balanced")
        result = cm.ravel()  # returns flattened array
        trues = result[0] + result[4] + result[8]
        runs2.append(trues)

    X_imp_train, X_imp_test, y_train, y_test = train_test_split(X_imp, y, test_size=0.25, random_state=1)
    runs3 = []
    test_weight={"Iris-setosa":0.8,"Iris-versicolor":1.1,"Iris-virginica":1.1}
    for solver in solvers:
        cm = logreg(X_imp_train, y_train, X_imp_test, y_test, solver, test_weight)
        result = cm.ravel()  # returns flattened array
        trues = result[0] + result[4] + result[8]
        runs3.append(trues)

    print("... Summary of model correct predictions ....\n", runs)
    print("... (Added class_weight balanced) Summary of model correct predictions ....\n", runs2)
    print("... (Added class_weight "+ str(test_weight) +") Summary of model correct predictions ....\n", runs3)

main()