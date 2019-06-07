# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 11:04:22 2019

@author: asad
"""

#Importing Required Libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt


#importing ML models from scikit learn package

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#Loading Dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

#Dimension of dataset
print(dataset.shape)

#Peek of data
print(dataset.head(20))

#Statistical Summary
print(dataset.describe())

#Class distribution
print(dataset.groupby('class').size())


#Visualization

##Univariet 
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False,figsize=(10,8))
plt.show()
#histogram 
dataset.hist(figsize=(10,8))
plt.show()

#Multivariet Plots
## scatter plot matrix
scatter_matrix(dataset,figsize=(12,10))
plt.show()

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, 
                                                                test_size=validation_size, 
                                                                random_state=seed)
# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

'''
Here you can se how I performed all the steps if we have cleaned data because
this all the procedure doesn't include cleaning or prepocessing part.
And such that we can find confusion matrics and classification report for all the algorithms.