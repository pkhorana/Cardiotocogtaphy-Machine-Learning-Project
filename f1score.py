#This program finds the F1 score of training each feature idndividually with the data to see which feature is the most important for making predictions.


from numpy import random
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
excel_file = r"C:\Users\pranav.khorana\Documents\Internship 2017-2018\Cardiotocography data\CTGexp.xls"
df = pd.read_excel(excel_file, sheet_name="Raw Data")
array = df.values

cart =  DecisionTreeClassifier()

print("CART Decision Tree")


number = 1
x = 0
y = 1
while y < 22:
    mean = 0
    print("Feature %d" % (number,))
    number+=1
    from sklearn.model_selection import StratifiedKFold
    X = array[:,x:y]
    Y = array[:,21]
    x+=1
    y+=1
    skf = StratifiedKFold(n_splits=10)
    for train_index, test_index in skf.split(X, Y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        Xtrain, Xvalidation = X[train_index], X[test_index]
        Ytrain, Yvalidation = Y[train_index], Y[test_index]
        random.seed(3)
        cart.fit(Xtrain, Ytrain)
        predictions = cart.predict(Xvalidation)
        mean += f1_score(Yvalidation, predictions, average='macro')
    print(mean/10)
    print("")
    print("")
    print("")
    print("")


