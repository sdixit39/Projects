import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("Data12.csv")
df.fillna(0,inplace=True)
print("df=\n",df)
Y=df['Station']
from sklearn.preprocessing import LabelEncoder
Y=LabelEncoder().fit_transform(Y)
#print("Y=",Y)
print("shape(y)=",np.shape(Y))

x1=df['Jan']
x2=df['Feb']
x3=df['Mar']
x4=df['Apr']
x5=df['May']
x6=df['June']
x7=df['July']
x8=df['Aug']
x9=df['Sep']
x10=df['Oct']
x11=df['Nov']
x12=df['Dec']

X=[x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12]
X=np.matrix(X)
X=np.transpose(X)

a=np.zeros((4187,12))

for i in range(4187):
        for j in range(12):
            if X[i,j]=='351,.6':
                 X[i,j]=351
            elif X[i,j]=='0-':
                 X[i,j]=0
            a[i,j]=float(X[i,j])
        #print(type(X[i,j]))
X=a  
print("shape(X)=",np.shape(X))  


plt.plot(Y)
#plt.show()

Max=np.max(Y)
print('max=',Max)
data=[]
data2=[]

k=0
for i in range(Max):
    for j in range(len(X)):
        if Y[j]==i:
            #print('OK')
            k=k+1
            data.append(X[j,:])


    data=np.reshape(data,(k,12))
    data=np.matrix(data)
    print('shape=',np.shape(data))
    am=np.mean(data,axis=0)       
    dataM=np.mean(data,axis=0)
    print('shape dataM=',np.shape(dataM))
    data=[]
    k=0
    data2.append(dataM)
    
#data2=np.reshape(data,(k,12))


data2=np.reshape(data2,(Max,12))
#data2=np.reshape(data2,(1116,2))
print('shape data2=',np.shape(data2))

#Convert to csv
'''
header_list=['1','2','3','4','5','6','7','8','9','10','11','12']

gf = pd.DataFrame(data2)
print(gf.head())
print(gf)
gf.to_csv('1.csv',header=header_list,index=False)
'''
y=dataM

X=[]
x1=[]

for i in range(93):
    for j in range(12):
        X=[i,j]
        x1.append(X)
print('x=',X)
print('shape=',np.shape(x1))

'''
header_list=['0','1']

gf = pd.DataFrame(x1)
print(gf.head())
print(gf)
gf.to_csv('2.csv',header=header_list,index=False)
'''

y=data2
'''
header_list=['1','2','3','4','5','6','7','8','9','10','11','12']

gf = pd.DataFrame(y)
print(gf.head())
print(gf)
gf.to_csv('3.csv',header=header_list,index=False)
'''



y=np.reshape(y,(1116,1))
print(y)
print(np.shape(y))

'''
header_list=['1']

gf = pd.DataFrame(y)
print(gf.head())
print(gf)
gf.to_csv('4.csv',header=header_list,index=False)
'''

#training-lr

print("\n-----------LR--------------")
from sklearn.linear_model import LinearRegression
mdl=LinearRegression()
mdl.fit(x1,y)
print("\nmdl.score(x1,y)=",mdl.score(x1,y)*100,"%")

#training-knn
print("\n-----------KNN--------------")
from sklearn.neighbors import KNeighborsClassifier
mdl=KNeighborsClassifier(n_neighbors=3)
y=np.ceil(y)
mdl.fit(x1,y)
print("\nmdl1.score(x1,y)=",mdl.score(x1,y)*100,"%")



#training-nb
print("\n-----------NB--------------")
from sklearn.naive_bayes import GaussianNB
mdl=GaussianNB()
mdl.fit(x1,y)
print("\nmdl.score(x1,y)=",mdl.score(x1,y)*100,"%")

#testing prediction
x11=int(input("enter district:"))
x22=int(input("enter month:"))
I=[x11,x22]
I=np.matrix(I)

result=mdl.predict(I)
result=np.floor(result)
print("result=",result)





       







































































