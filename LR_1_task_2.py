# ПР-1 Юшко ТЦР-43 — Завдання 2.2. Попередня обробка нових даних

import numpy as np
from sklearn import preprocessing

VARIANT = 30
values = [-5.3, -8.9, 4.2, 2.9, -5.0, -3.3, 3.1, -2.8, -3.2, 4.2, -1.4, 6.1]
threshold = 1.5

input_data_v = np.array(values).reshape(4, 3)
print("Variant", VARIANT, "\nInput data:\n", input_data_v, "\nThreshold =", threshold)

# Бінаризація
data_binarized_v = preprocessing.Binarizer(threshold=threshold).transform(input_data_v)
print("\nBinarized data:\n", data_binarized_v)

# Виключення середнього
print("\nBEFORE: ")
print("Mean =", input_data_v.mean(axis=0))
print("Std deviation =", input_data_v.std(axis=0))
data_scaled_v = preprocessing.scale(input_data_v)
print("\nAFTER: ")
print("Mean =", data_scaled_v.mean(axis=0))
print("Std deviation =", data_scaled_v.std(axis=0))

# Масштабування MinMax
data_scaled_minmax_v = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(input_data_v)
print("\nMin max scaled data:\n", data_scaled_minmax_v)

# Нормалізація
data_normalized_l1_v = preprocessing.normalize(input_data_v, norm='l1')
data_normalized_l2_v = preprocessing.normalize(input_data_v, norm='l2')
print("\nl1 normalized data:\n", data_normalized_l1_v)
print("\nl2 normalized data:\n", data_normalized_l2_v)
