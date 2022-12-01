"""
Authors: Zuzanna Ciborowska s20682 & Joanna Walkiewicz s20161
Python project to create decision tree and svm classificator for Pima Indians Diabetes
and Card Fraud Transactions Dataset
System requirements:
- Python 3.10
- Pandas
- Pydotplus
- IPython
- Sckit-learn
- Graphviz
"""
import pandas as pd
import pydotplus as pdp

from IPython.display import Image
from six import StringIO
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

"""
Loading the datasets: one with Pima Indians Diabetes data, and second with card transaction data
"""
dataPima = pd.read_csv('pima-indians-diabetes.csv')
dataCard = pd.read_csv('card_transdata_smaller.csv')

data_cols_pima = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction',
              'Age']
data_cols_card = ['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price',
             'repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']

"""
Splitting Pima Indians Diabetes data for 67% training and 33% test
"""
x = dataPima[data_cols_pima]
y = dataPima.Outcome
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=1)

"""
Crating classifier for decision tree
"""
def tree(X_train, Y_train):
    classifier = DecisionTreeClassifier()
    classifier = classifier.fit(X_train, Y_train)

    return classifier

#pima indians
"""
Creating and making image of decision tree for Pima Indians Diabetes data
"""
classifier = tree(X_train, Y_train)

dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names=data_cols_pima, class_names=['0', '1'])
graph = pdp.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())

"""
Create a svm Classifier for Pima Indians Diabetes data
"""
svc = svm.SVC(kernel='rbf', C=1, gamma=100).fit(x, y)

"""
Train the model using the training sets
"""
svc.fit(X_train, Y_train)

"""
Predict the response for test dataset
"""
y_pred = svc.predict(X_test)

"""
Model Accuracy: how often is the classifier correct?
"""
print("Accuracy for Pima Indians Diabetes data:",metrics.accuracy_score(Y_test, y_pred))


#card transaction
"""
Splitting Card transaction data for 67% training and 33% test
"""
x = dataCard[data_cols_card]
y = dataCard.fraud
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=1)

"""
Creating and making image of decision tree for Card transaction data
"""
classifier = tree(X_train, Y_train)

dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names=data_cols_card, class_names=['0', '1'])
graph = pdp.graph_from_dot_data(dot_data.getvalue())
graph.write_png('fraud.png')
Image(graph.create_png())

"""
Create a svm Classifier for Card transaction data
"""
svc = svm.SVC(kernel='rbf', C=1, gamma=100).fit(x, y)

"""
Train the model using the training sets
"""
svc.fit(X_train, Y_train)

"""
Predict the response for test dataset
"""
y_pred = svc.predict(X_test)

"""
Model Accuracy: how often is the classifier correct?
"""
print("Accuracy for Card transaction data:",metrics.accuracy_score(Y_test, y_pred))
