# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Use the standard libraries in python for finding linear regression.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Predict the values of array.
5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
6.Obtain the graph.

## Program:

Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: G.TEJASWINI
RegisterNumber:  212222230157

```python

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X=data[:, [0, 1]]
y=data[:, 2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot, sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return j,grad
  
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)

x_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)

def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  return j
def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-y)/X.shape[0]
  return grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
  y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  X_plot=np.c_[xx.ravel(),yy.ravel()]
  X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot=np.dot(X_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
  plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
  
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta, X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob >= 0.5).astype(int)

np.mean(predict(res.x,X)==y)
```

## Output:

1.Array of X
             ![image](https://github.com/TejaswiniGugananthan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121222763/4a153249-82de-4922-9bc3-2cfb3f840784)

2.Array Value of y
             ![image](https://github.com/TejaswiniGugananthan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121222763/9c3bcdb4-1f38-4633-8989-ec3c4377f162)

3.Exam 1 - score graph
            <img width="282" alt="image" src="https://github.com/TejaswiniGugananthan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121222763/ce073978-3311-4cfa-87ea-4a754149050c">

4.Sigmoid function graph
            <img width="281" alt="image" src="https://github.com/TejaswiniGugananthan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121222763/e6521cce-5589-42a6-b61d-cb0a24122f4d">

5.X_train_grad value
             <img width="114" alt="image" src="https://github.com/TejaswiniGugananthan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121222763/aec8109b-539f-4e04-829f-e87095d83f5a">

6.Y_train_grad value
          <img width="97" alt="image" src="https://github.com/TejaswiniGugananthan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121222763/1291c7c3-b8ca-4e3f-8168-a41053f2a205">

7.Print res.x
          <img width="155" alt="image" src="https://github.com/TejaswiniGugananthan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121222763/d37124d7-bd3a-47e0-ad9b-0d1c07ad72aa">

8.Decision boundary - graph for exam score
           <img width="276" alt="image" src="https://github.com/TejaswiniGugananthan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121222763/d82d197e-41e4-4e37-9f71-eb4fd83f3985">

9.Proability value
           <img width="105" alt="image" src="https://github.com/TejaswiniGugananthan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121222763/0e185156-4d6b-4330-892a-25ae866b22d3">

10.Prediction value of mean
           <img width="42" alt="image" src="https://github.com/TejaswiniGugananthan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121222763/96154aa6-5c19-45b6-928b-c083e6b04806">




## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

