#This program makes predictions based on the F1 scores of each feature.
#The purpose is to take out the less important features one by one to see if the performance measures improve.

from numpy import random
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
excel_file = r"C:\Users\pranav.khorana\Documents\Internship 2017-2018\Cardiotocography data\CTGexp.xls"
df = pd.read_excel(excel_file, sheet_name="Raw Data")
array = df.values
mean = 0
cart =  DecisionTreeClassifier()
lr = LogisticRegression()
knn = KNeighborsClassifier()

number = 20

print("%d Features" % (number,))
print("")
print("")
count = 1
from sklearn.model_selection import StratifiedKFold
X1 = array[:,0:7]
X2 = array[:,9:21]
X = np.column_stack((X1,X2))
y = array[:,21]
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
    print("")
    print("")
    print("")
    print("")
    mean += accuracy_score(Yvalidation, predictions)
number -= 1
mean = mean/10
print("look")
print (mean)
mean = 0


print("%d Features" % (number,))
print("")
print("")
count = 1
X1 = array[:,0:8]
X2 = array[:,9:14]
X3 = array[:,15:21]
X = np.column_stack((X1,X2,X3))
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
    print("")
    print("")
    print("")
    print("")
    mean += accuracy_score(Yvalidation, predictions)
number -= 1
mean = mean / 10
print("look")
print(mean)
mean = 0


print("%d Features" % (number,))
print("")
print("")
count = 1
X1 = array[:,0:8]
X2 = array[:,9:14]
X3 = array[:,16:21]
X = np.column_stack((X1,X2, X3))
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
    print("")
    print("")
    print("")
    print("")
    mean += accuracy_score(Yvalidation, predictions)
number -= 1
mean = mean / 10
print("look")
print(mean)
mean = 0



print("%d Features" % (number,))
print("")
print("")
count = 1
X1 = array[:,0:8]
X2 = array[:,9:14]
X3 = array[:,16:20]
X = np.column_stack((X1, X2, X3))
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
    print("")
    print("")
    print("")
    print("")
    mean += accuracy_score(Yvalidation, predictions)
number -= 1
mean = mean / 10
print("look")
print(mean)
mean = 0




print("%d Features" % (number,))
print("")
print("")
count = 1
X1 = array[:,0:1]
X2 = array[:,2:8]
X3 = array[:,9:14]
X4 = array[:,16:20]
X = np.column_stack((X1, X2, X3, X4))
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
    print("")
    print("")
    print("")
    print("")
    mean += accuracy_score(Yvalidation, predictions)
number -= 1
mean = mean / 10
print("look")
print(mean)
mean = 0




print("%d Features" % (number,))
print("")
print("")
count = 1
X1 = array[:,0:1]
X2 = array[:,2:3]
X3 = array[:,4:8]
X4 = array[:,9:14]
X5 = array[:,16:20]
X = np.column_stack((X1, X2, X3, X4, X5))
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
    print("")
    print("")
    print("")
    print("")
    mean += accuracy_score(Yvalidation, predictions)
number -= 1
mean = mean / 10
print("look")
print(mean)
mean = 0



print("%d Features" % (number,))
print("")
print("")
count = 1
X1 = array[:,0:1]
X2 = array[:,2:3]
X3 = array[:,4:8]
X4 = array[:,9:13]
X5 = array[:,16:20]
X = np.column_stack((X1, X2, X3, X4, X5))
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
    print("")
    print("")
    print("")
    print("")
    mean += accuracy_score(Yvalidation, predictions)
number -= 1
mean = mean / 10
print("look")
print(mean)
mean = 0



print("%d Features" % (number,))
print("")
print("")
count = 1
X1 = array[:,2:3]
X2 = array[:,4:8]
X3 = array[:,9:13]
X4 = array[:,16:20]
X = np.column_stack((X1, X2, X3, X4))
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
    print("")
    print("")
    print("")
    mean += accuracy_score(Yvalidation, predictions)
number -= 1
mean = mean / 10
print("look")
print(mean)
mean = 0





print("%d Features" % (number,))
print("")
print("")
count = 1
X1 = array[:,4:8]
X2 = array[:,9:13]
X3 = array[:,16:20]

X = np.column_stack((X1, X2, X3))
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
    print("")
    print("")
    print("")
    print("")
    mean += accuracy_score(Yvalidation, predictions)
number -= 1
mean = mean / 10
print("look")
print(mean)
mean = 0




print("%d Features" % (number,))
print("")
print("")
count = 1
X1 = array[:,4:8]
X2 = array[:,10:13]
X3 = array[:,16:20]

X = np.column_stack((X1, X2, X3))
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
    print("")
    print("")
    print("")
    print("")
    mean += accuracy_score(Yvalidation, predictions)
number -= 1
mean = mean / 10
print("look")
print(mean)
mean = 0





print("%d Features" % (number,))
print("")
print("")
count = 1
X1 = array[:,4:7]
X2 = array[:,10:13]
X3 = array[:,16:20]

X = np.column_stack((X1, X2, X3))
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
    print("")
    print("")
    print("")
    print("")
    mean += accuracy_score(Yvalidation, predictions)
number -= 1
mean = mean / 10
print("look")
print(mean)
mean = 0



print("%d Features" % (number,))
print("")
print("")
count = 1
X1 = array[:,4:7]
X2 = array[:,10:12]
X3 = array[:,16:20]

X = np.column_stack((X1, X2, X3))
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
    print("")
    print("")
    print("")
    print("")
    mean += accuracy_score(Yvalidation, predictions)
number -= 1
mean = mean / 10
print("look")
print(mean)
mean = 0




print("%d Features" % (number,))
print("")
print("")
count = 1
X1 = array[:,4:7]
X2 = array[:,10:12]
X3 = array[:,16:18]
X4 = array[:,19:20]

X = np.column_stack((X1, X2, X3, X4))
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
    print("")
    print("")
    print("")
    print("")
    mean += accuracy_score(Yvalidation, predictions)
number -= 1
mean = mean / 10
print("look")
print(mean)
mean = 0


print("%d Features" % (number,))
print("")
print("")
count = 1
X1 = array[:,4:7]
X2 = array[:,10:12]
X3 = array[:,17:18]
X4 = array[:,19:20]

X = np.column_stack((X1, X2, X3, X4))
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
    print("")
    print("")
    print("")
    print("")
    mean += accuracy_score(Yvalidation, predictions)
number -= 1
mean = mean / 10
print("look")
print(mean)
mean = 0




print("%d Features" % (number,))
print("")
print("")
count = 1
X1 = array[:,4:7]
X2 = array[:,10:11]
X3 = array[:,17:18]
X4 = array[:,19:20]

X = np.column_stack((X1, X2, X3, X4))
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
    print("")
    print("")
    print("")
    print("")
    mean += accuracy_score(Yvalidation, predictions)
number -= 1
mean = mean / 10
print("look")
print(mean)
mean = 0



print("%d Features" % (number,))
print("")
print("")
count = 1
X1 = array[:,4:7]
X3 = array[:,17:18]
X4 = array[:,19:20]

X = np.column_stack((X1, X2, X3))
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
    print("")
    print("")
    print("")
    print("")
    mean += accuracy_score(Yvalidation, predictions)
number -= 1
mean = mean / 10
print("look")
print(mean)
mean = 0





print("%d Features" % (number,))
print("")
print("")
count = 1
X1 = array[:,4:7]
X2 = array[:,17:18]

X = np.column_stack((X1, X2))
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
    print("")
    print("")
    print("")
    print("")
    mean += accuracy_score(Yvalidation, predictions)
number -= 1
mean = mean / 10
print("look")
print(mean)
mean = 0




print("%d Features" % (number,))
print("")
print("")
count = 1
X1 = array[:,4:5]
X2 = array[:,6:7]
X3 = array[:,17:18]

X = np.column_stack((X1, X2, X3))
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
    print("")
    print("")
    print("")
    print("")
    mean += accuracy_score(Yvalidation, predictions)
number -= 1
mean = mean / 10
print("look")
print(mean)
mean = 0



print("%d Features" % (number,))
print("")
print("")
count = 1
X1 = array[:,4:5]
X2 = array[:,6:7]

X = np.column_stack((X1, X2))
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
    print("")
    print("")
    print("")
    print("")
    mean += accuracy_score(Yvalidation, predictions)
number -= 1
mean = mean / 10
print("look")
print(mean)
mean = 0



print("%d Features" % (number,))
print("")
print("")
count = 1
X1 = array[:,6:7]
X = np.column_stack((X1, X2))
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
    print("")
    print("")
    print("")
    print("")
    mean += accuracy_score(Yvalidation, predictions)
number -= 1
mean = mean / 10
print("look")
print(mean)
mean = 0




