#This program creates an equal number of data for pathlogical, suspicious, and normal cases so that the data is balanced.

from numpy import random
import random
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

cart =  DecisionTreeClassifier()
lr = LogisticRegression()
knn = KNeighborsClassifier()

count = 1
from sklearn.model_selection import StratifiedKFold
# X = array[:,0:21]
# y = array[:,21]

normal = []
susp = []
patho = []


# normal = [0][0]
# susp = [0][0]
# patho = [0][0]



for i in range(0, 2126):
    if array[i,21] == 1:
        normal.append(array[i,:])
    elif array[i,21] == 2:
        susp.append(array[i, :])
    elif array[i,21] == 3:
        patho.append(array[i, :])

print(len(normal))
print(len(susp))
print(len(patho))

arrayList = []
for x in range(300):
  arrayList.append(normal[random.randint(0, len(normal)-1)])

for x in range(len(susp)):
    arrayList.append(susp[x])
for x in range(300-len(susp)):
  arrayList.append(susp[random.randint(0, len(susp)-1)])

for x in range(len(patho)):
    arrayList.append(patho[x])
for x in range(300-len(patho)):
  arrayList.append(patho[random.randint(0, len(patho)-1)])



array = np.asarray(arrayList)

print(array.shape)

from sklearn.model_selection import StratifiedKFold
X = array[:,0:21]
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
