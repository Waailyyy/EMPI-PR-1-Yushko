# ПР-1 Юшко ТЦР-43 — Завдання 2.3. Логістичний класифікатор

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from utilities import visualize_classifier

X = np.array([[3.1, 7.2], [4, 6.7], [2.9, 8], [5.1, 4.5],
              [6, 5], [5.6, 5], [3.3, 0.4],
              [3.9, 0.9], [2.8, 1],
              [0.5, 3.4], [1, 4], [0.6, 4.9]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

# liblinear працює за схемою «один проти всіх»; у нових версіях sklearn її треба задавати явно
from sklearn.multiclass import OneVsRestClassifier
classifier = OneVsRestClassifier(linear_model.LogisticRegression(solver='liblinear', C=1))
classifier.fit(X, y)
print("Predicted:", classifier.predict(X))
print("Actual:   ", y)
print("Training accuracy =", round(100 * (classifier.predict(X) == y).mean(), 2), "%")

visualize_classifier(classifier, X, y)
