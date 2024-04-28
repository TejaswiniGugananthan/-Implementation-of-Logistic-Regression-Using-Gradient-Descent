# Exp 5 Implementation-of-Logistic-Regression-Using-Gradient-Descent
### NAME: G.TEJASWINI
### Register Number: 212222230157
## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary.
6. Define a function to predict the 
   Regression value.

## Program:

### Program to implement the Logistic Regression Using Gradient Descent.

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```
```
dataset = pd. read_csv('Placement_Data.csv')
dataset
```
### Dataset
![image](https://github.com/Leann4468/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121165979/3e6fce21-22eb-4523-949f-8451e99eb6fd)

```
dataset = dataset.drop('sl_no', axis=1)
dataset = dataset.drop('salary', axis=1)
```
```
dataset ["gender"] = dataset ["gender"]. astype ('category')
dataset["ssc_b"] = dataset["ssc_b"].astype( 'category')
dataset ["hsc_b"] = dataset ["hsc_b"].astype ('category')
dataset ["degree_t"] = dataset ["degree_t"].astype ('category' )
dataset ["workex"] = dataset ["workex"].astype ('category')
dataset ["specialisation"] = dataset ["specialisation"].astype ('category')
dataset["status"] = dataset["status"].astype ('category')
dataset ["hsc_s"] = dataset["hsc_s"].astype( 'category')
dataset.dtypes
```
### Dataset.dtypes
![image](https://github.com/Leann4468/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121165979/b374ae0a-a578-4739-9eb8-58b99578a256)

```
dataset[ "gender"] = dataset ["gender"].cat.codes
dataset["ssc_b"] = dataset ["ssc_b"].cat.codes
dataset["hsc_b"] = dataset ["hsc_b"].cat.codes
dataset ["degree_t"] = dataset ["degree_t"].cat.codes
dataset ["workex"] = dataset ["workex"].cat.codes
dataset["specialisation"] = dataset ["specialisation"].cat.codes
dataset ["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset ["hsc_s"].cat.codes
dataset
```
### Dataset
![image](https://github.com/Leann4468/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121165979/50987e2c-673d-4d18-8eba-8cb9fe70adcf)

```
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
```
```
Y
```
### Y
![image](https://github.com/Leann4468/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121165979/b75aae0f-9dd4-4050-a5e8-865a0a2201ac)

```
theta = np.random.randn(X. shape[1])
y=Y
```
```
def sigmoid(z):
    return 1 / (1 + np. exp(-z) )
```
```
def loss(theta,X,y):
    h=sigmoid(x.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
```
```
def gradient_descent(theta,X,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot(h-y)/m
        theta-=alpha*gradient
    return theta
```
```
theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)
```
```
def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred 
```
```
y_pred=predict(theta,X)
```
```
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
```
### Accuracy
![image](https://github.com/Leann4468/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121165979/9df20f7c-2348-4b27-98cd-d05e3dbbe651)

```
print(y_pred)
```
### Y_pred
![image](https://github.com/Leann4468/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121165979/76218537-0f95-4bc1-abfa-e07749fd910c)
```
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```
### Y_prednew
![image](https://github.com/Leann4468/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121165979/133b5e29-494e-4def-ac32-252f6f01d69b)
```
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```
### Y_prednew
![image](https://github.com/Leann4468/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121165979/088468d4-4fef-4a77-9742-13b386691f6d)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
