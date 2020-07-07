import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
import wget

#------------------------- Downloading Data ---------------------------
#url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/teleCust1000t.csv'
#teleCust1000t = wget.download(url)

#----- print all columns
pd.set_option('display.max_columns', None)

#--------------------- Reading data -----------------------------------
df = pd.read_csv('teleCust1000t.csv')
print(df.head())


#--------------------- Data Visualization and Analysis ----------------
print(df['custcat'].value_counts())
df.hist(column='income', bins=50)
plt.show()

#--------------------- Define feature set -----------------------------
print(df.columns)
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values
print(X[0:5])

y = df['custcat'].values
print(y[0:5])

#--------------------- Normalize Data ----------------------------------
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
print(X[0:5])

#--------------------- Train test split --------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


#--------------------- Classification -----------------------------------
from sklearn.neighbors import KNeighborsClassifier

#----- training with k=4
k = 4
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
print(neigh)
yhat = neigh.predict(X_test)
print(yhat[0:5])

#-------------------- Accucracy evaluation -------------------------------
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


#----- training with k=6
k = 6
neigh6 = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
yhat6 = neigh6.predict(X_test)
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh6.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat6))

#---------------  Calculate the accuracy of KNN for different Ks ----------
Ks = 15
mean_acc = np.zeros((Ks - 1))
std_acc = np.zeros((Ks - 1))
ConfustionMx = [];
for n in range(1, Ks):
    # train model and predict
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n - 1] = metrics.accuracy_score(y_test, yhat)
    std_acc[n - 1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])
print(mean_acc)

#------------- Plot model accuracy for different number of neighbors -------
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()
plt.clf()
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)