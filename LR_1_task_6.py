# ПР-1 Юшко ТЦР-43 — Завдання 2.6. Машина опорних векторів (SVM)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from utilities import visualize_classifier
from sklearn.metrics import confusion_matrix

data = np.loadtxt('data_multivar_nb.txt', delimiter=',')
X, y = data[:, :-1], data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

models = {
    'Naive Bayes (GaussianNB)': GaussianNB(),
    'SVM, linear kernel': SVC(kernel='linear', C=1.0),
    'SVM, RBF kernel': SVC(kernel='rbf', C=1.0, gamma='scale'),
}
rows = []
for name, model in models.items():
    train_acc = 100 * (model.fit(X, y).predict(X) == y).mean()
    model.fit(X_train, y_train)
    test_acc = 100 * (model.predict(X_test) == y_test).mean()
    cv = {s: 100 * cross_val_score(model, X, y, scoring=s, cv=3).mean()
          for s in ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']}
    rows.append([name, round(train_acc, 2), round(test_acc, 2)] + [round(v, 2) for v in cv.values()])

results = pd.DataFrame(rows, columns=['Model', 'Full-data acc, %', 'Test acc, %', 'CV Accuracy, %', 'CV Precision, %', 'CV Recall, %', 'CV F1, %'])
print(results.to_string(index=False))

svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_classifier.fit(X_train, y_train)
y_test_pred = svm_classifier.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_test_pred, digits=3))
print("Confusion matrix:\n", confusion_matrix(y_test, y_test_pred))
visualize_classifier(svm_classifier, X_test, y_test)
