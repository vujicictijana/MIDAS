import pandas as pd
from os import walk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import time
from sklearn.metrics import precision_recall_fscore_support as score
import bz2
import _pickle as cPickle
import numpy as np


my_os = "mac" #win mac

separator = "\\"

if my_os == "mac":
    separator = "//"


path = 'Dataset' + separator;

files = []

for (dirpath, dirnames, filenames) in walk(path):
    files.extend(filenames)
    break

n = len(files)
finalDF = pd.read_csv(path + files[0],sep = ';', dtype=str)

for i in range(1,n):
    if files[i].endswith('.csv'):
        df = pd.read_csv(path + files[i],  sep = ';', dtype=str)
        finalDF = pd.concat([finalDF, df])

print("Total rows: " , finalDF.shape[0])

print("RUNS total: " + str(len(finalDF['Run id'].unique())))

anomalies = finalDF[(finalDF['Anomaly'] == '1') | (finalDF['Anomaly'] == '2') | (finalDF['Anomaly'] == '3')  ]

print("RUNS with anomalies:" + str(len(anomalies['Run id'].unique())))

X = finalDF.drop(['Timestamp', 'Parameter for Anomaly','Anomaly','Actual value','Run id'],axis= 1)
X = X.replace(',','.', regex=True)
X = X.astype(float)
# print(X)
print("Total columns X: " , X.shape[1])

Y = finalDF['Anomaly'];

# print(Y.unique())

Y = Y.replace('-',0);
Y = Y.replace(float('nan'),0);

Y = pd.to_numeric(Y)

print("Classes:", Y.unique())

normal = Y <1
normal_arr = Y[normal]

print("Normal instances: " , len(normal_arr))

anomaly = Y >=1
anomaly_arr = Y[anomaly]

print("Anomaly instances: " , len(anomaly_arr))
print("Anomaly instances - class 1 (step): " , np.count_nonzero(Y == 1))
print("Anomaly instances - class 2 (freeze): " , np.count_nonzero(Y == 2))
print("Anomaly instances - class 3 (ramp): " , np.count_nonzero(Y == 3))

# binary or multiclass
# binary classification - anomaly detection
# multiclass classification - anomaly classification
classification  = "binary"
if classification  == "binary":
    Y = Y.replace(2,1);
    Y = Y.replace(3,1);

# create single csv
# newDF = pd.concat([X, Y], axis=1)
# newDF.to_csv("all.csv")
# print(newDF)

# normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X.values), columns=X.columns, index=X.index)

# split train-validation-test 70%-10%-20%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.35, random_state=1)

print("Train: ", len(y_train))
print("Test: ", len(y_test))
print("Validation: ", len(y_val))
print("Classes:", y_train.unique())

# Decision Tree - DT
from sklearn import tree

start_time = time.time()
result = ""
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
accuracy = accuracy_score(prediction, y_test)
print("DT: ", accuracy)
# print(classification_report(y_test, prediction))

my_time = (time.time() - start_time)

precision, recall, fscore, support = score(y_test, prediction)

cm = confusion_matrix(y_test, prediction)

result = "DT:\nAccuracy: " +  str(accuracy) +"\nPrecision: " + str(precision) + "\nrecall: " + str(recall)+ "\nfscore: " + str(fscore) + "\nsupport: " + str(support) + "\nConfusion matrix:\n" + str(cm) +  "\nTime (s): " + str(my_time)
f1 = open("DT.txt", "w")
f1.write(result)
f1.close()
# print("--- %s seconds ---" % (time.time() - start_time))

f = bz2.BZ2File("DT.pbz2", "wb")
cPickle.dump((clf),f)
f.close()

# Logistic Regression - LR

from sklearn.linear_model import LogisticRegression
start_time = time.time()
result = ""
logisticRegr = LogisticRegression(max_iter=1000)
logisticRegr.fit(X_train, y_train)
prediction = logisticRegr.predict(X_test)
accuracy = accuracy_score(prediction, y_test)
print("LR: ", accuracy)

my_time = (time.time() - start_time)

precision, recall, fscore, support = score(y_test, prediction)
cm = confusion_matrix(y_test, prediction)

result = "LR:\nAccuracy: " +  str(accuracy) +"\nPrecision: " + str(precision) + "\nrecall: " + str(recall)+ "\nfscore: " + str(fscore) + "\nsupport: " + str(support) + "\nConfusion matrix:\n" + str(cm) +   "\nTime (s): " + str(my_time)
f1 = open("LR.txt", "w")
f1.write(result)
f1.close()
# print(classification_report(y_test, prediction))
f = bz2.BZ2File("LR.pbz2", "wb")
cPickle.dump((logisticRegr),f)
f.close()

# Random Forest - RF

from sklearn.ensemble import RandomForestClassifier

x = range(5, 50, 5)
max = 0
maxAcc = 0
for n in x:
    rf = RandomForestClassifier(n_estimators = n)
    rf.fit(X_train, y_train)
    prediction = rf.predict(X_val)
    accuracy = accuracy_score(prediction, y_val)
    print("No. of trees: ", n)
    print("Accuracy: ", accuracy)
    if maxAcc<accuracy:
        max = n
        maxAcc = accuracy


print("Best no. of trees: ", max)
print("Best accuracy: ", maxAcc)

start_time = time.time()
result = ""

rf = RandomForestClassifier(n_estimators = max)
rf.fit(X_train, y_train)
prediction = rf.predict(X_test)
accuracy = accuracy_score(prediction, y_test)
print("RF: ", accuracy)


my_time = (time.time() - start_time)

precision, recall, fscore, support = score(y_test, prediction)
cm = confusion_matrix(y_test, prediction)

result = "RF:\n" + "Best no. of trees: " + str(max) + "\nAccuracy: " +  str(accuracy) +"\nPrecision: " + str(precision) + "\nrecall: " + str(recall)+ "\nfscore: " + str(fscore) + "\nConfusion matrix:\n" + str(cm) +  "\nsupport: " + str(support) +  "\nTime (s): " + str(my_time)
f1 = open("RF.txt", "w")
f1.write(result)
f1.close()

f = bz2.BZ2File("RF.pbz2", "wb")
cPickle.dump((rf),f)
f.close()

# K Nearest Neighbors - KNN

from sklearn.neighbors import KNeighborsClassifier

x = range(1, 20, 2)
max = 0
maxAcc = 0
for n in x:
    knn = KNeighborsClassifier(n_neighbors=n)
    print("K: ", n)
    knn.fit(X_train, y_train)
    prediction = knn.predict(X_val)
    accuracy = accuracy_score(prediction, y_val)
    print("Accuracy: ", accuracy)
    if maxAcc<accuracy:
        max = n
        maxAcc = accuracy

print("Best K: ", max)
print("Best accuracy: ", maxAcc)

start_time = time.time()
result = ""

knn = KNeighborsClassifier(n_neighbors=max)
knn.fit(X_train, y_train)
prediction = knn.predict(X_test)
accuracy = accuracy_score(prediction, y_test)
print("KNN: ", accuracy)

my_time = (time.time() - start_time)

precision, recall, fscore, support = score(y_test, prediction)
cm = confusion_matrix(y_test, prediction)

result = "KNN:\n" + "Best K: " + str(max) + "\nAccuracy: " +  str(accuracy) +"\nPrecision: " + str(precision) + "\nrecall: " + str(recall)+ "\nfscore: " + str(fscore) + "\nsupport: " + str(support) + "\nConfusion matrix:\n" + str(cm) +   "\nTime (s): " + str(my_time)
f1 = open("KNN.txt", "w")
f1.write(result)
f1.close()

f = bz2.BZ2File("KNN.pbz2", "wb")
cPickle.dump((knn),f)
f.close()

# Multilayer Perceptron - MLP (relu activation function)

from sklearn.neural_network import MLPClassifier

x = range(5, 105, 5)
max = 0
maxAcc = 0
for n in x:
    mlp = MLPClassifier(hidden_layer_sizes=(n),activation="relu" , max_iter=2000)
    mlp.fit(X_train, y_train)
    prediction = mlp.predict(X_val)
    accuracy = accuracy_score(prediction, y_val)
    print("No. of neurons: ", n)
    print("Accuracy: ", accuracy)
    if maxAcc<accuracy:
        max = n
        maxAcc = accuracy

print("Best n: ", max)
print("Best accuracy: ", maxAcc)

start_time = time.time()
result = ""

mlp = MLPClassifier(hidden_layer_sizes=(max),activation="relu" , max_iter=2000)
mlp.fit(X_train, y_train)
prediction = mlp.predict(X_test)
accuracy = accuracy_score(prediction, y_test)
print("MLP relu: ", accuracy)

my_time = (time.time() - start_time)

precision, recall, fscore, support = score(y_test, prediction)
cm = confusion_matrix(y_test, prediction)

result = "MLP relu:\n" + "Best n: " + str(max) + "\nAccuracy: " +  str(accuracy) +"\nPrecision: " + str(precision) + "\nrecall: " + str(recall)+ "\nfscore: " + str(fscore) + "\nsupport: " + str(support) + "\nConfusion matrix:\n" + str(cm) +  "\nTime (s): " + str(my_time)
f1 = open("MLP relu.txt", "w")
f1.write(result)
f1.close()

f = bz2.BZ2File("MLP relu.pbz2", "wb")
cPickle.dump((mlp),f)
f.close()

# Multilayer Perceptron - MLP (tanh activation function)

from sklearn.neural_network import MLPClassifier

x = range(5, 105, 5)
max = 0
maxAcc = 0
for n in x:
    mlp = MLPClassifier(hidden_layer_sizes=(n),activation="tanh" , max_iter=2000)
    mlp.fit(X_train, y_train)
    prediction = mlp.predict(X_val)
    accuracy = accuracy_score(prediction, y_val)
    print("No. of neurons: ", n)
    print("Accuracy: ", accuracy)
    if maxAcc<accuracy:
        max = n
        maxAcc = accuracy

print("Best n: ", max)
print("Best accuracy: ", maxAcc)

start_time = time.time()
result = ""

mlp = MLPClassifier(hidden_layer_sizes=(max),activation="tanh" , max_iter=2000)
mlp.fit(X_train, y_train)
prediction = mlp.predict(X_test)
accuracy = accuracy_score(prediction, y_test)
print("MLP tanh: ", accuracy)

my_time = (time.time() - start_time)

precision, recall, fscore, support = score(y_test, prediction)
cm = confusion_matrix(y_test, prediction)

result = "MLP tanh:\n" + "Best n: " + str(max) + "\nAccuracy: " +  str(accuracy) +"\nPrecision: " + str(precision) + "\nrecall: " + str(recall)+ "\nfscore: " + str(fscore) + "\nsupport: " + str(support) + "\nConfusion matrix:\n" + str(cm) +   "\nTime (s): " + str(my_time)
f1 = open("MLP tanh.txt", "w")
f1.write(result)
f1.close()

f = bz2.BZ2File("MLP tanh.pbz2", "wb")
cPickle.dump((mlp),f)
f.close()

# Multilayer Perceptron - MLP (logistic activation function)

from sklearn.neural_network import MLPClassifier

x = range(5, 105, 5)
max = 0
maxAcc = 0
for n in x:
    mlp = MLPClassifier(hidden_layer_sizes=(n),activation="logistic" , max_iter=2000)
    mlp.fit(X_train, y_train)
    prediction = mlp.predict(X_val)
    accuracy = accuracy_score(prediction, y_val)
    print("No. of neurons: ", n)
    print("Accuracy: ", accuracy)
    if maxAcc<accuracy:
        max = n
        maxAcc = accuracy

print("Best n: ", max)
print("Best accuracy: ", maxAcc)

start_time = time.time()
result = ""

mlp = MLPClassifier(hidden_layer_sizes=(max),activation="logistic" , max_iter=2000)
mlp.fit(X_train, y_train)
prediction = mlp.predict(X_test)
accuracy = accuracy_score(prediction, y_test)
print("MLP logistic: ", accuracy)

my_time = (time.time() - start_time)

precision, recall, fscore, support = score(y_test, prediction)
cm = confusion_matrix(y_test, prediction)

result = "MLP logistic:\n" + "Best n: " + str(max) + "\nAccuracy: " +  str(accuracy) +"\nPrecision: " + str(precision) + "\nrecall: " + str(recall)+ "\nfscore: " + str(fscore) + "\nsupport: " + str(support) +"\nConfusion matrix:\n" + str(cm) +    "\nTime (s): " + str(my_time)
f1 = open("MLP logistic.txt", "w")
f1.write(result)
f1.close()

f = bz2.BZ2File("MLP logistic.pbz2", "wb")
cPickle.dump((mlp),f)
f.close()

# Support Vector Machine - SVM (linear kernel)

from sklearn import svm
start_time = time.time()
result = ""
svm = svm.SVC(kernel='linear')
svm.fit(X_train, y_train)
prediction = svm.predict(X_test)
accuracy = accuracy_score(prediction, y_test)
print("SVM linear: ", accuracy)
my_time = (time.time() - start_time)

precision, recall, fscore, support = score(y_test, prediction)
cm = confusion_matrix(y_test, prediction)

result = "SVM linear:\nAccuracy: " +  str(accuracy) +"\nPrecision: " + str(precision) + "\nrecall: " + str(recall)+ "\nfscore: " + str(fscore) + "\nsupport: " + str(support) +"\nConfusion matrix:\n" + str(cm) +   "\nTime (s): " + str(my_time)
f1 = open("SVM linear.txt", "w")
f1.write(result)
f1.close()

f = bz2.BZ2File("SVM linear.pbz2", "wb")
cPickle.dump((svm),f)
f.close()

# Support Vector Machine - SVM (poly kernel)

from sklearn import svm
start_time = time.time()
result = ""
svm = svm.SVC(kernel='poly')
svm.fit(X_train, y_train)
prediction = svm.predict(X_test)
accuracy = accuracy_score(prediction, y_test)
print("SVM poly: ", accuracy)
my_time = (time.time() - start_time)

precision, recall, fscore, support = score(y_test, prediction)
cm = confusion_matrix(y_test, prediction)

result = "SVM poly:\nAccuracy: " +  str(accuracy) +"\nPrecision: " + str(precision) + "\nrecall: " + str(recall)+ "\nfscore: " + str(fscore) + "\nsupport: " + str(support) +"\nConfusion matrix:\n" + str(cm) +   "\nTime (s): " + str(my_time)
f1 = open("SVM poly.txt", "w")
f1.write(result)
f1.close()

f = bz2.BZ2File("SVM poly.pbz2", "wb")
cPickle.dump((svm),f)
f.close()

# Support Vector Machine - SVM (sigmoid kernel)

from sklearn import svm
start_time = time.time()
result = ""
svm = svm.SVC(kernel='sigmoid')
svm.fit(X_train, y_train)
prediction = svm.predict(X_test)
accuracy = accuracy_score(prediction, y_test)
print("SVM sigmoid: ", accuracy)
my_time = (time.time() - start_time)

precision, recall, fscore, support = score(y_test, prediction)
cm = confusion_matrix(y_test, prediction)

result = "SVM sigmoid:\nAccuracy: " +  str(accuracy) +"\nPrecision: " + str(precision) + "\nrecall: " + str(recall)+ "\nfscore: " + str(fscore) + "\nsupport: " + str(support) +"\nConfusion matrix:\n" + str(cm) +   "\nTime (s): " + str(my_time)
f1 = open("SVM sigmoid.txt", "w")
f1.write(result)
f1.close()

f = bz2.BZ2File("SVM sigmoid.pbz2", "wb")
cPickle.dump((svm),f)
f.close()


# Support Vector Machine - SVM (rbf kernel)

from sklearn import svm
start_time = time.time()
result = ""
svm = svm.SVC(kernel='rbf')
svm.fit(X_train, y_train)
prediction = svm.predict(X_test)
accuracy = accuracy_score(prediction, y_test)
print("SVM rbf: ", accuracy)
my_time = (time.time() - start_time)

precision, recall, fscore, support = score(y_test, prediction)
cm = confusion_matrix(y_test, prediction)

result = "SVM rbf:\nAccuracy: " +  str(accuracy) +"\nPrecision: " + str(precision) + "\nrecall: " + str(recall)+ "\nfscore: " + str(fscore) + "\nsupport: " + str(support) +"\nConfusion matrix:\n" + str(cm) +   "\nTime (s): " + str(my_time)
f1 = open("SVM rbf.txt", "w")
f1.write(result)
f1.close()

f = bz2.BZ2File("SVM rbf.pbz2", "wb")
cPickle.dump((svm),f)
f.close()