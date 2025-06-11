import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

X=np.zeros((505,10000))
y=np.zeros(505)

for i in range(1,505):
    f='FAce/FAce/1 ('+str(i)+')'+'.jpg'
    I=cv2.imread(f)
    I=cv2.resize(I,(100,100))

    I=cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
    #print('shape=',np.shape(I))

    I=np.reshape(I,(1,10000))
    I=np.matrix(I)
    
    X[i-1,:]=I
    y[i-1]=i
    
print('shape of x=',np.shape(X))
print('shape of y=',np.shape(y))

#print('x=',X)
#print('y=',y)



'''
print("\n-----------LR--------------")

from sklearn.linear_model import LinearRegression
mdl=LinearRegression()
mdl.fit(X,y)
print("\nmdl.score(x,y)=",mdl.score(X,y)*100,"%")


print("\n-----------KNN--------------")
from sklearn.neighbors import KNeighborsClassifier
mdl=KNeighborsClassifier(n_neighbors=3)
mdl.fit(X,y)
print("\nmdl1.score(x,y)=",mdl.score(X,y)*100,"%")
'''
print("\n-----------NB--------------")
from sklearn.naive_bayes import GaussianNB
mdl=GaussianNB()
mdl.fit(X,y)
print("\nmdl.score(x,y)=",mdl.score(X,y)*100,"%")


I=cv2.imread('FAce/FAce/1 (500).jpg')
#cv2.imshow('img',I)
I=cv2.resize(I,(100,100))
I=cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
print('shape=',np.shape(I))

I=np.reshape(I,(1,10000))
I=np.matrix(I)
print('shape=',np.shape(I))

result=mdl.predict(I)
result=np.floor(result)
print("result = ",result)


df=pd.read_csv('emotion level.csv')
df=np.matrix(df)
print(len(df))
print('shape=',np.shape(df))

y1=df[:,1]
y1=np.array(y1)
print('shape=',np.shape(y1))
result=int(result)

if 0<=result<=72:
    print("emotion : Angry")
    print("emotion level : ",y1[result-1])
elif 73<=result<=144:
    print("Disgust")
    print("emotion level : ",y1[result-1])
elif 145<=result<=216:
    print("emotion : Fear")
    print("emotion level : ",y1[result-1])
elif 217<=result<=288:
    print("emotion : Happy")
    print("emotion level : ",y1[result-1])
elif 289<=result<=360:
    print("emotion : Neutral")
    print("emotion level : ",y1[result-1])
elif 361<=result<=432:
    print("emotion : Sad")
    print("emotion level : ",y1[result-1])
elif 433<=result<=504:
    print("emotion : Surprise")
    print("emotion level : ",y1[result-1])
