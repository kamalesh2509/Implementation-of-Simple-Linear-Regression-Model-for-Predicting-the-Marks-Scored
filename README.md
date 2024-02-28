# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph. 
5. Predict the regression for marks by using the representation of the graph.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: E Kamalesh
RegisterNumber:  212222100019
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```


## Output:
### df.head()
![229978451-2b6bdc4f-522e-473e-ae2f-84ec824344c5](https://github.com/kamalesh2509/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120444689/52176394-5f83-4828-8273-f1b685194c2f)

### df.tail()
 ![229978854-6af7d9e9-537f-4820-a10b-ab537f3d0683](https://github.com/kamalesh2509/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120444689/813ad97d-cc2b-4187-8ed0-02eabb4cc3ca)
### Array value of X
![image](https://github.com/kamalesh2509/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120444689/6090dc76-f478-4820-aacf-b47f728adbdd)
### Array value of Y
![image](https://github.com/kamalesh2509/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120444689/6d174060-62f3-43f5-9370-634f614358f9)
### Values of Y prediction
![image](https://github.com/kamalesh2509/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120444689/5f0de00a-3ba9-49e4-bb67-c66528cecaca)
### Array values of Y test
![image](https://github.com/kamalesh2509/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120444689/49023daa-c38c-44e5-9927-2ea1adb442e3)

### Training Set Graph
![229979169-ad4db5b6-e238-4d80-ae5b-405638820d35](https://github.com/kamalesh2509/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120444689/1ffbdc7c-0564-442a-895b-257632e99778)
### Test Set Graph

![229979225-ba90853c-7fe0-4fb2-8454-a6a0b921bdc1](https://github.com/kamalesh2509/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120444689/a2007149-2ec0-4d17-bc79-d5376583871b)
### Values of MSE, MAE and RMSE


![229979276-bb9ffc68-25f8-42fe-9f2a-d7187753aa1c](https://github.com/kamalesh2509/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120444689/5fdda121-2198-42fb-8efe-747c689d9726)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
