#!/usr/bin/python3
#Thesis Kamil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def main():
	data = pd.read_csv('features3.csv')
	#print(data.head())

	# Separating the independent variables from dependent variables
	x=data.iloc[:,:-1]
	y=data.iloc[:,4]
	x_train,x_test, y_train, y_test=train_test_split(x,y,test_size=0.25)

	# Fitting Random Forest Classification to the Training set
	classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
	classifier.fit(x_train, y_train)

	# Predicting the Test set results
	y_pred = classifier.predict(x_test)

	accuracy = accuracy_score(y_test,y_pred)
	print(accuracy)




if __name__ =='__main__':
    main()