#!/usr/bin/python3
#Thesis Kamil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

def main():
	data = pd.read_csv('features1.csv')
	#print(data.head())

	# Separating the independent variables from dependent variables
	x=data.iloc[:,:-1]
	y=data.iloc[:,155]
	x_train,x_test, y_train, y_test=train_test_split(x,y,test_size=0.20)

	model=SVC()
	model.fit(x_train, y_train)

	pred=model.predict(x_test)

	print(confusion_matrix(y_test,pred))
	print(classification_report(y_test, pred))
	print(accuracy_score(y_test,pred))

	#print(x)
	#print(y.shape)






if __name__ =='__main__':
    main()