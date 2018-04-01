import numpy as np
from numpy import random
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from random import randrange
#use pandas to load the data
excel_file = r"C:\Users\pranav.khorana\Documents\Internship 2017-2018\Cardiotocography data\CTGexp.xls"
df = pd.read_excel(excel_file, sheet_name="Raw Data")
array = df.values

cart =  DecisionTreeClassifier()
lr = LogisticRegression()
knn = KNeighborsClassifier()

count = 1
from sklearn.model_selection import StratifiedKFold
X = array[:,0:22]
y = array[:,22]
skf = StratifiedKFold(n_splits=10)
for train_index, test_index in skf.split(X, y):
    # print("TRAIN:", train_index, "TEST:", test_index)
    Xtrain, Xvalidation = X[train_index], X[test_index]
    Ytrain, Yvalidation = y[train_index], y[test_index]

    random.seed(3)


    print("Test %d" % (count,))
    count+=1
    # split out validation data-set

    print("CART Decision Tree")
    cart.fit(Xtrain, Ytrain)
    predictions = cart.predict(Xvalidation)
    print(accuracy_score(Yvalidation, predictions))
    print(confusion_matrix(Yvalidation, predictions))
    print(classification_report(Yvalidation, predictions))
    print("")
    print("")

    print("KNN")
    knn = KNeighborsClassifier()
    knn.fit(Xtrain, Ytrain)
    predictions = knn.predict(Xvalidation)
    print(accuracy_score(Yvalidation, predictions))
    print(confusion_matrix(Yvalidation, predictions))
    print(classification_report(Yvalidation, predictions))
    print("")
    print("")

    print("Logistic Regression")
    lr = LogisticRegression()
    lr.fit(Xtrain, Ytrain)
    predictions = lr.predict(Xvalidation)
    print(accuracy_score(Yvalidation, predictions))
    print(confusion_matrix(Yvalidation, predictions))
    print(classification_report(Yvalidation, predictions))
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")


