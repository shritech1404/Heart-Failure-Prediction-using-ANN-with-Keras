import pandas as pd
import matplotlib.pyplot as plt

heart = pd.read_csv("heart_failure_clinical_records_dataset.csv")

X = heart.drop('DEATH_EVENT', axis='columns')
y = heart.DEATH_EVENT

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()

model.add(Dense(units=3, kernel_initializer='he_uniform', activation='relu', input_dim=12))
model.add(Dense(units=3, kernel_initializer='he_uniform', activation='relu'))

model.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))

model.compile(optimizer='adamax', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_split=0.33, batch_size=10, epochs=500)

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

model.predict(75.0, 0, 582, 0, 20, 1, 265000.00, 1.9, 130, 1, 0, 4)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)

model.summary()