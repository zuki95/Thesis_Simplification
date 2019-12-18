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

def main():
	data = pd.read_csv('features5.csv')
	#print(data.head())

	# Separating the independent variables from dependent variables
	x=data.iloc[:,:-1]
	y=data.iloc[:,5]
	x_train,x_test, y_train, y_test=train_test_split(x,y,test_size=0.25)

	dtree_model = DecisionTreeClassifier(max_depth = 2).fit(x_train, y_train) 
	dtree_predictions = dtree_model.predict(x_test)

	# creating a confusion matrix 
	cm = confusion_matrix(y_test, dtree_predictions)
	print(cm) 

	ac =  accuracy_score(y_test,dtree_predictions)

	print(ac)




if __name__ =='__main__':
    main()