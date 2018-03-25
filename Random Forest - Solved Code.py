# Import Basic Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import data from the excel downloaded
CC_data = pd.read_excel('default_of_credit_card_clients.xlsx','Data')
CC_data.shape
CC_data.info()
df = CC_data[1:]

X=df.drop(labels=['Y'],axis=1)

X.info()
y=(CC_data[1:].Y).astype('int')


# Data Cleaning and Data Transformations (Redying for Model consumption)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()


# Transforming the Pay status columns

X['X6']=labelencoder_X.fit_transform(X['X6'])
X['X7']=labelencoder_X.fit_transform(X['X7'])

X['X8']=labelencoder_X.fit_transform(X['X8'])

X['X9']=labelencoder_X.fit_transform(X['X9'])

X['X10']=labelencoder_X.fit_transform(X['X10'])
X['X11']=labelencoder_X.fit_transform(X['X11'])



# Transforming Catagorical variables with onehot encoding

onehotencoder = OneHotEncoder(categorical_features=[2,3])
X = onehotencoder.fit_transform(X).toarray()
X = np.delete (X , 7, axis=1)
X = X [:,1:]


# Splitting the data into Train set and Test Set for training the model

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Importing Random Forest Model and building the Model

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=500)
rfc.fit(X_train,y_train)
y1 = rfc.predict(X_test)


# In[37]:

from sklearn.metrics import confusion_matrix
cnf=confusion_matrix(y_test,y1)
print(cnf)


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = rfc, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()


#Predicted probabilities of defaulter for x_test
rfc.predict_proba(X_test)
Pred_prob=pd.DataFrame({'Pred_prob':list(rfc.predict_proba(X_test)[:,1])})
X=pd.DataFrame(X_test)
X['pred']=Pred_prob
type(Pred_prob)

X['defaulter']=pd.DataFrame({'Pred_prob':list(y_test)})


# Sort the values and prepare the data for Sorting Smoothing Method

X=X.sort_values('pred',axis=0)
X = X.reset_index(drop=True)
y_sort=X['defaulter']
len(y_test)


# Sorting Smoothing Method (SSM) to estimate the real probability of default

X['act']=0
actual_prob=[]    
#sms
for i in range(0,len(X)):
    sum=0
    count=0
    n=50
    for j in range(i-n,i+n):
        if (j>=0 and j<len(X)):
            sum=sum+X['defaulter'][j]
            count=count+1
    val=sum/count
    actual_prob.append(val)


# In[45]: Transforming the SSM list to Dataframe

act_prob1=pd.DataFrame({'Pred_prob':actual_prob})
X['act']=act_prob1  
    
X1=X['act']
y1=X['pred']
X.info()
X1.shape


# Importing Linear Regression to Predict accuracy of probability of default

from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(X[['pred']],X[['act']])   
results=pd.DataFrame(lr.coef_)
y2=lr.predict(X[['pred']])


# Importing Metrics to measure error and accuracy

from sklearn import metrics

score=metrics.r2_score(y1,X1)
MSE = metrics.mean_squared_error(X1,y1)


# Plot the "Preicted vs Actual Default probability" graph to show the linearity 

plt.scatter(X1,y1)
plt.plot(X1.reshape(-1,1), (lr.intercept_+lr.predict(X1.reshape(-1,1))*lr.coef_) , color = 'red')
plt.title('Preicted vs Actual Default probability (Test set)')
plt.xlabel('Predicted Credit Score')
plt.ylabel('Actual Credit Score')
plt.show()
print(lr.intercept_, lr.coef_, score, MSE)
# [0.01059746], [0.89108218], 0.943629602242, 0.00229062613953


# In the predictive accuracy of probability of default, Random Forest show the performance based on R2 (0.9436, close to 1), and regression intercept (0.0106, close to 0) and negligible error of 0.0022