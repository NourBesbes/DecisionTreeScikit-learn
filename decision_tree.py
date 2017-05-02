import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.cross_validation import train_test_split
import json

train_df = pd.read_csv("./BA.csv")
y = targets = labels = train_df["sport"].values
columns = ["temperature", "heartbeatting", "tension"]
X=features = train_df[list(columns)].values
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.3, random_state = 100)

clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf = clf.fit(X_train, y_train)

with open("./tree.dot", 'w') as f:
  f = tree.export_graphviz(clf, out_file=f, feature_names=columns)

#Prediction
print(clf.predict([[7.25, 3.,0.]]))
#cross_validation
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
modele_all= lr.fit(X_train, y_train)
print(modele_all.coef_,modele_all.intercept_)
from sklearn import cross_validation
scores= cross_validation.cross_val_score(lr,X_train, y_train,cv=10,scoring='accuracy')
print("Accuracy per fold: ")
print(scores)
print("Average accuracy: ", scores.mean())

#module pour evaluation des classifieurs
from sklearn import metrics
#fonction pour evaluation de methodes
def error_rate(modele,y_test,X_test):
	#prediction
	y_train = modele.predict(X_test)
	#taux d erreur
	err = 1.0 -metrics.accuracy_score(y_test,y_train)
	#return
	return err
	#fin fonction
print "Error",(error_rate(clf,y_test,X_test))


#rules
def rules(clf, features, labels, node_index=0):
    """Structure of rules in a fit decision tree classifier

    Parameters
    ----------
    clf : DecisionTreeClassifier
        A tree that has already been fit.

    features, labels : lists of str
        The names of the features and labels, respectively.

    """
    node = {}
    if clf.tree_.children_left[node_index] == -1:  # indicates leaf
        count_labels = zip(clf.tree_.value[node_index, 0], labels)
        node['name'] = ', '.join(('{} of {}'.format(int(count), label)
                                  for count, label in count_labels))
    else:
        feature = features[clf.tree_.feature[node_index]]
        threshold = clf.tree_.threshold[node_index]
        node['name'] = '{} > {}'.format(feature, threshold)
        left_index = clf.tree_.children_left[node_index]
        right_index = clf.tree_.children_right[node_index]
        node['children'] = [rules(clf, features, labels, right_index),
                            rules(clf, features, labels, left_index)]
    return node

r=rules(clf,columns, ['sport'])
with open('rules.json', 'w') as f:
	f.write(json.dumps(r))

#Cross validation schema

import matplotlib.pyplot as plt
# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt

lr2 = linear_model.LinearRegression()
fig,ax = plt.subplots()
predicted= cross_val_predict(lr2,X_train, y_train,cv=10)
print predicted
ax.scatter(y_train,predicted)
ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
