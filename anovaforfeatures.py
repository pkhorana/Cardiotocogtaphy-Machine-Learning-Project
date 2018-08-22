from numpy import random
from scipy import stats
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



for i in range(0, 2126):
    if array[i,21] == 1:
        normal.append(array[i,:])
    elif array[i,21] == 2:
        susp.append(array[i, :])
    elif array[i,21] == 3:
        patho.append(array[i, :])

normal = np.asarray(normal)
susp = np.asarray(susp)
patho = np.asarray(patho)

number = 1
x = 0

while x < 21:
    mean = 0
    print("Feature %d" % (number,))
    number+=1
    print(stats.f_oneway(normal[:,x], susp[:,x], patho[:,x]))
    x+=1



