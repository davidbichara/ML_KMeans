from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import pandas as pd


bc = load_breast_cancer()

X = bc.data
y = bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#amount of clusters on graph, random_state zero makes the randomness the same each time
model = KMeans(n_clusters=2, random_state=0)

#model below does not include y because it is clustering solely off of X
model.fit(X_train)
predictions = model.predict(X_test)

labels = model.labels_
print("labels ", labels)
print("predictions ", predictions)
print("accuracy ", accuracy_score(y_test, predictions))