#importing libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

# importing the dataset
data = pd.read_csv("/content/data.csv")
data.head()

#checking the size of data
data.shape

# getting types of columns
data.columns

# just checking that is there any null values in dataset or not
data.isnull().sum()

# getting basic description of the dataset
data.describe()

# visualising the data
sns.relplot(x='price',y='bedrooms',data=data)

sns.relplot(x='price',y='bathrooms',data=data)

sns.relplot(x='price',y='sqft_living',hue='waterfront',data=data)

# building the model
data.head(5)

#importing the libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# droping down the data columns which are unnessasory for the algorithm and our result
train = data.drop(['date','floors','waterfront','view','street','city','statezip','country'],axis=1)
test = data['price']

# splitting the dataset in train and testing dataset
X_train,X_test,y_train,y_test = train_test_split(train, test, test_size=0.3, random_state=2)

# training the model using - sklearn
regr = LinearRegression()

# fitting our train dataset to machine
regr.fit(X_train,y_train)

#lets predict from dataset left for testing
pred = regr.predict(X_test)
pred

# efficiency of the model 
regr.score(X_test,y_test)

#plot the tested data set
plt.ylabel("price")
plt.xlabel("parameters")
plt.plot(X_test,y_test)
# its a linear data woohooo...

#plotting the efficiency of the model
plt.scatter(y_test,pred)

