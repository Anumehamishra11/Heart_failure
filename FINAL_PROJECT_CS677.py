#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn import metrics
from sklearn.svm import SVC
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,classification_report,roc_curve,auc,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from warnings import simplefilter
pd.options.mode.chained_assignment = None
from sklearn.model_selection import train_test_split
simplefilter(action='ignore', category=FutureWarning)
from yellowbrick.classifier import ClassificationReport
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA


# In[313]:


pip install yellowbrick


# In[3]:


input_dir = os.getcwd()
labels = os.path.join(input_dir, 'heart_failure.csv')
data = pd.read_csv(labels)
data.head()


# In[4]:


data.info()


# In[5]:


data.isnull().sum()


# In[6]:


def plot_hist(col, bins=30, title="",xlabel="",ax=None):
    sns.distplot(col, bins=bins,ax=ax)
    ax.set_title(f'Histogram of {title}',fontsize=20)
    ax.set_xlabel(xlabel)


# In[7]:


## Numerical categories 
fig, axes = plt.subplots(3,2,figsize=(20,20),constrained_layout=True)
plot_hist(data.creatinine_phosphokinase,
          title='Histogram of Creatinine Phosphokinase',
          xlabel="Level of the CPK (mcg/L)",
          ax=axes[0,0])
plot_hist(data.platelets,
          bins=30,
          title='Histogram of Platelets',
          xlabel='Platelets in the blood (kiloplatelets/mL)',
          ax=axes[0,1])
plot_hist(data.serum_creatinine,
          title='Histogram of Serum Creatinine', 
          xlabel='Level of serum creatinine in the blood (mg/dL)',
          ax=axes[1,0])
plot_hist(data.serum_sodium,
          bins=30,
          title='Histogram of Serum Sodium',
          xlabel='Level of serum sodium in the blood (mEq/L)',
          ax=axes[1,1])
plot_hist(data.ejection_fraction,
          title='Histogram of Ejection Fraction', 
          xlabel='Percentage of blood leaving the heart at each contraction (percentage)',
          ax=axes[2,0])
plot_hist(data.time,
          bins=30,
          title='Histogram of Time',
          xlabel='Follow-up period (days)',
          ax=axes[2,1])
plt.show()


# In[8]:


#Distribution of age among Death Events 
fig1 = px.histogram(data, x="age",color="DEATH_EVENT")
fig1.show()


# In[9]:


#Distribution of sex among Death Events 
fig2 = px.histogram(data, x="sex",color="DEATH_EVENT")
fig2.show()


# In[10]:


#Total number of male and females in dataset
women = len(data[data["sex"]==0])
men = len(data[data["sex"]==1])


# In[11]:


#Men survived and dead
dead_men = len(data.loc[(data["DEATH_EVENT"]==1) &(data['sex']==1)])
alive_men = men - dead_men
#Women survived and dead 
dead_women = len(data.loc[(data["DEATH_EVENT"]==1) & (data['sex']==0)])
alive_women = women - dead_women


# In[12]:


labels = ['Men died','Men survived','Women died','Women survived']
values = [dead_men, alive_men, dead_women, alive_women]
fig = go.Figure(data=[go.Pie(labels=labels, values=values,textinfo='label+percent',hole=0.4)])
fig.update_layout(
    title_text="Distribution of DEATH EVENT according to their gender")
fig.show()


# In[13]:


sns.heatmap(data.corr(),cmap="Greys");


# In[14]:


pd.crosstab(data.diabetes ,data.DEATH_EVENT).plot(kind='bar')
plt.legend(title='DEATH_EVENT', loc='upper right', labels=['No death event', 'Death event'])
plt.title('Death Event VS  diabetes ')
plt.xlabel('Diabetes ')
plt.ylabel('Death')
plt.show()


# In[15]:


pd.crosstab(data.high_blood_pressure ,data.DEATH_EVENT).plot(kind='bar')
plt.legend(title='DEATH_EVENT', loc='upper right', labels=['Not alive', 'Alive'])
plt.title('Death Event VS High pressure blood ')
plt.xlabel('High pressure blood ')
plt.ylabel('Death')
plt.show()


# In[16]:


pd.crosstab(data.smoking ,data.DEATH_EVENT).plot(kind='bar')
plt.legend(title='DEATH_EVENT', loc='upper right', labels=['Not alive', 'Alive'])
plt.title('Death Event VS smokers ')
plt.xlabel('Smokers ')
plt.ylabel('Death')
plt.show()


# In[17]:


pd.crosstab(data.anaemia ,data.DEATH_EVENT).plot(kind='bar')
plt.legend(title='DEATH_EVENT', loc='upper right', labels=['No death event', 'Death event'])
plt.title('Death Event VS anaemia ')
plt.xlabel('Anaemia ')
plt.ylabel('Death')
plt.show()


# In[18]:


pd.crosstab(data.sex ,data.DEATH_EVENT).plot(kind='bar')
plt.legend(title='DEATH_EVENT', loc='upper right', labels=['No death event', 'Death event'])
plt.title('Death Event VS Age ')
plt.xlabel('Age ')
plt.ylabel('Death')
plt.show()


# In[19]:


#Possible scenerios for a given patient according to their age
fig = px.box(data, x="DEATH_EVENT", y="age", color="smoking", notched=True)
fig.show()


# In[20]:


#Possible scenerios for a given patient according to their sex
fig = px.box(data, x="DEATH_EVENT", y="age", color="sex", notched=True)
fig.show()


# In[21]:


# select all columns except 'DEATH_EVENT'
cols = [col for col in data.columns if col not in ['DEATH_EVENT']]
# dropping the DEATH_EVENT column
data2 = data[cols]
#assigning the DEATH_EVENT column as result
result = data['DEATH_EVENT']
#split data set into train and test sets
x_train, x_test, y_train , y_test = train_test_split(data2 ,result, test_size = 0.20, random_state = 10)


# In[22]:


## Logistic Regression
# Scaling my testing and training data
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
scaler.fit(x_test)
x_test = scaler.transform(x_test)
# Create Logistic Regression classifer object
LR = LogisticRegression()
# Train the algorithm on training data 
model = LR.fit(x_train, y_train)
# Predicting using the testing data
y_pred = model.predict(x_test)
# Accuracy score of the model
print("Accuracy using Logistic Regression:",round(metrics.accuracy_score(y_test, y_pred)*100,2),"%")
# Instantiate the classification model and visualizer
visualizer = ClassificationReport(model, classes=['1','0'])
visualizer.fit(x_train, y_train)  # Fit the training data to the visualizer
visualizer.score(x_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             


# In[22]:


#ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve ({auc(fpr, tpr):.2f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=500, height=400)
fig.show()

#Confusion Matrix
print("Confusion matrix for Logistic Regression is:"'\n',
      confusion_matrix(y_test,y_pred))


# In[23]:


##GaussianNB
x_train, x_test, y_train , y_test = train_test_split(data2 ,result, test_size = 0.20, random_state = 10)
#Create an object of the type GaussianNB
gnb = GaussianNB()
#Scaling my testing and training data
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
scaler.fit(x_test)
x_test = scaler.transform(x_test)
#Train the algorithm on training data and predict using the testing data
pred = gnb.fit(x_train, y_train).predict(x_test)
#Accuracy score of the model
print("Accuracy using Naive-Bayes : ",round(accuracy_score(y_test, pred, normalize = True)*100,2),"%")
# Instantiate the classification model and visualizer
visualizer = ClassificationReport(gnb, classes=['1','0'])
visualizer.fit(x_train, y_train)  # Fit the training data to the visualizer
visualizer.score(x_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof() 


# In[24]:


#ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve ({auc(fpr, tpr):.2f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=500, height=400)
fig.show()
matrix_NB = confusion_matrix(y_test,y_pred)
#Confusion Matrix
print("Confusion matrix for Naive-Bayes is:"'\n',matrix_NB)


# In[28]:


## Linear SVM
x_train, x_test, y_train , y_test = train_test_split(data2 ,result, test_size = 0.20, random_state = 10)
# Scaling my testing and training data
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
scaler.fit(x_test)
x_test = scaler.transform(x_test)
# Create an object of type Linear SVM
svm_model = svm.SVC(kernel='linear')
# Train the algorithm on training data and predict using the testing data
pred = svm_model.fit(x_train, y_train).predict(x_test)
# Accuracy of the model
print("Accuracy using Linear SVM : ",round(accuracy_score(y_test, pred, normalize = True)*100,2),"%")
# Instantiate the classification model and visualizer
visualizer = ClassificationReport(svm_model, classes=['1','0'])
visualizer.fit(x_train,y_train)  # Fit the training data to the visualizer
visualizer.score(x_test,y_test)  # Evaluate the model on the test data
g = visualizer.poof()   


# In[27]:


#ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve ({auc(fpr, tpr):.2f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=500, height=400)
fig.show()
matrix_SVM = confusion_matrix(y_test,y_pred)
#Confusion Matrix
print("Confusion matrix for  Linear SVM is:"'\n', matrix_SVM)


# In[34]:


## Random Forest Classifier
x_train, x_test, y_train , y_test = train_test_split(data2 ,result, test_size = 0.20, random_state = 10)
# Create an object of Random Forest Classifier
RFC = RandomForestClassifier()
# Scaling my data 
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
scaler.fit(x_test)
x_test = scaler.transform(x_test)
# Train RandomForest Classifer on training dataset
model = RFC.fit(x_train,y_train)
# Predicting the response for testing dataset
y_pred = model.predict(x_test)
# Accuracy of the model
print("Accuracy using random forest:",round(metrics.accuracy_score(y_test, y_pred)*100,2),"%")
# Instantiate the classification model and visualizer
visualizer = ClassificationReport(model, classes=['1','0'])
visualizer.fit(x_train,y_train)  # Fit the training data to the visualizer
visualizer.score(x_test,y_test)  # Evaluate the model on the test data
g = visualizer.poof()             


# In[35]:


#ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve ({auc(fpr, tpr):.2f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=500, height=400)
fig.show()

#Confusion Matrix
print("Confusion matrix for Random Forest Classifier is:"'\n',
      confusion_matrix(y_test,y_pred))


# In[36]:


## Decision Tree Classifier 
x_train, x_test, y_train , y_test = train_test_split(data2 ,result, test_size = 0.20, random_state = 10)
# Scaling my testing and training data
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
scaler.fit(x_test)
x_test = scaler.transform(x_test)
# Train Decision Tree Classifier 
classifier = DecisionTreeClassifier(criterion = 'entropy')
model = classifier.fit(x_train,y_train)
# Predict the response for test dataset
y_pred = classifier.predict(x_test)
# Model Accuracy
print("Accuracy using Decision Tree classifier :",round(metrics.accuracy_score(y_test, y_pred)*100,2),"%")
# Instantiate the classification model and visualizer
visualizer = ClassificationReport(model, classes=['1','0'])
visualizer.fit(x_train,y_train)  # Fit the training data to the visualizer
visualizer.score(x_test,y_test)  # Evaluate the model on the test data
g = visualizer.poof()     


# In[37]:


#ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve ({auc(fpr, tpr):.2f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=500, height=400)
fig.show()

#Confusion Matrix
print("Confusion matrix for Decision Tree Classifier is:"'\n',
      confusion_matrix(y_test,y_pred))


# In[38]:


x = data.copy()
y = x.loc[:,["DEATH_EVENT"]]
x = x.drop(columns=['time','DEATH_EVENT'])
features_names = x.columns


# In[39]:


#Feature Selection 
forest = ExtraTreesClassifier(n_estimators=250,random_state=0)
forest.fit(x,y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")
for f in range(x.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


# In[33]:


# Plot the impurity-based feature importances of the forest
plt.figure()
plt.title("Importance of features")
sns.barplot(x=features_names[indices].to_numpy(), y=importances[indices], palette="deep",yerr=std[indices])
plt.xticks(range(x.shape[1]), features_names[indices].to_numpy(),rotation=80)
plt.xlim([-1, x.shape[1]])
plt.show()


# In[ ]:




