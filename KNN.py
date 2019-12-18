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

def main():
	data = pd.read_csv('features6.csv')
	#print(data.head())

	# Separating the independent variables from dependent variables
	x=data.iloc[:,:-1]
	y=data.iloc[:,4]
	x_train,x_test, y_train, y_test=train_test_split(x,y,test_size=0.25)

	knn = KNeighborsClassifier(n_neighbors = 7).fit(x_train, y_train)
	accuracy = knn.score(x_test, y_test) 

	print(accuracy)

	# creating a confusion matrix 
	knn_predictions = knn.predict(x_test)  
	cm = confusion_matrix(y_test, knn_predictions) 
	print(cm)



if __name__ =='__main__':
    main()