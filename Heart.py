# heart_disease
import numpy as np
import pandas as pd


import warnings
import pickle
warnings.filterwarnings("ignore")

#Read the dataset
heart=pd.read_csv('heart_disease.csv')

#Check the null values in the dataset
heart.isnull().sum()

#count the number or index
heart.index

# Missing value Handling
heart.dropna(inplace=True)

#Check the null values in the dataset
heart.isnull().sum()


# Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#Count the number of columns
len(heart.columns)

#Create the input and output variable
X = heart.iloc[:,0:14]  #independent columns
y = heart.iloc[:,15]    #target column i.e price range
y=y.astype('int')
#apply SelectKBest class to extract top 10 best features

bestfeatures = SelectKBest(score_func=chi2, k=5)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(5,'Score'))  #print 3 best features


#Create the input variable according to there score
l=['sysBP','age','totChol','cigsPerDay','diaBP']
x=heart[l].values
#Create the output variable
y=heart['TenYearCHD'].values


#Split the data set into train and test datasets
test_size=665
x_train=x[:-test_size]
y_train=y[:-test_size]

x_test=x[-test_size:]
y_test=y[-test_size:]




#Create Decision Tree model for predication 
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(x_train,y_train)

pickle.dump(dtc,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))



print("Accuracy Of Model :",dtc.score(x_test,y_test))
print("Predication of model :",dtc.predict([[106,39,195,0,70]]))
print("Predication of model :",dtc.predict([[142,46,294,0,94]]))
print("Predication of model :",dtc.predict([[140,38,221,20,70]]))




'''#Create Linear Regression model for predication 
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
print("Accuracy Of Model :",lr.score(x_test,y_test))
print("Predication of model :",lr.predict([[106,39,195,0,70]])) '''










































































# Model creation
l=['sysBP','age','totChol']
train=heart[l]
target=heart['TenYearCHD']
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(train,target)
print("Accuracy Of Model :",lr.score(train,target))
print("Predication of model :",lr.predict([[106,39,195]]))


from sklearn.linear_model import LogisticRegression
lor=LogisticRegression()
lor.fit(train,target)
print("Accuracy Of Model :",lor.score(train,target))
print("Predication of model :",lor.predict([[106,39,195]]))



from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(train,target)
print("Accuracy Of Model :",dtc.score(train,target))
print("Predication of model :",dtc.predict([[106,39,195]]))

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(train,target)
print("Accuracy Of Model :",rfc.score(train,target))
print("Predication of model :",rfc.predict([[106,39,195]]))

from sklearn.neighbors import KNeighborsClassifier
knc=KNeighborsClassifier()
knc.fit(train,target)
print("Accuracy Of Model :",knc.score(train,target))
print("Predication of model :",knc.predict([[106,39,195]]))

from sklearn.svm import SVC
sc=SVC()
sc.fit(train,target)
print("Accuracy Of Model :",sc.score(train,target))
print("Predication of model :",sc.predict([[106,39,195]]))


# confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = sc.predict(train)
print(confusion_matrix(target, pred))
print(classification_report(target, pred))

