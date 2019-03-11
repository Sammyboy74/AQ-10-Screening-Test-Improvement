#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 12:24:02 2018

@author: jessicarichardson
"""

# ALL DATA PROCESSING DONE IN SAS
# Packages needed for all data analysis
import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import time 
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz 
import pydotplus
import graphviz

# Reading in autism data set
autism = pd.read_csv("/Users/SamCoplin/Downloads/AUTISMVARS_FinalTAKE2.csv")
autism_labels = pd.read_csv("/Users/SamCoplin/Downloads/AUTISMLABELS_FinalTAKE2.csv")

# Splitting data set into test and train data set  (80%/20%)
X_train, X_test, y_train, y_test = train_test_split(autism, autism_labels, test_size=0.2, random_state = 1234)

# Analysis to complete: Logistic Regression, Decision Trees, 
# Random Forest, FSA, Neural Networks

### /////////////////////////////////////////////////////////////////////////
### ///////////////////////// Logistic Regression //////////////////////////
### ///////////////////////////////////////////////////////////////////////

start_time = time.time()
# Fitting a logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
oddsratios = np.vstack((list(X_train),np.exp(logreg.coef_)))

# Making predictions
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
print("--- %s seconds ---" % (time.time() - start_time))
# Plotting the ROC Curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


### /////////////////////////////////////////////////////////////////////////
### /////////////////////////// Decision Trees /////////////////////////////
### ///////////////////////////////////////////////////////////////////////

# Creating a function called Misclassplot that will input the four sets 
# of data and a depth and then plot the misclassification vs. depth of the 
# tree for both the training and testing data set. 

def DECISION_TREES(train, trainlab, test, testlab, depth, data):

    # Running a for loop to create data frames of the misclassification errors
    mis_train=[] # Initializing training misclassification list
    mis_test=[] # Initializing testing misclassification list
    for x in range(1,depth+1):
        dec_tree=DecisionTreeClassifier(max_depth=x) # Max_Depth attribute
        dec_tree=dec_tree.fit(train, trainlab) # Growing out the decision tree
        
        # Predicting on training data set / calculating misclassification error
        train_pred=dec_tree.predict(train)
        mis_train.append(100- (accuracy_score(trainlab,train_pred)*100))
        
        # Predicting on testing data set / calculating misclassification error
        test_pred=dec_tree.predict(test)
        mis_test.append(100- (accuracy_score(testlab,test_pred)*100))
        
    # Putting the misclassification errors into data frames
    mis_train = pd.DataFrame(mis_train)
    mis_train.columns=['Error']
    print(mis_train)
    mis_test = pd.DataFrame(mis_test)
    mis_test.columns = ['Error']
    print(mis_test)
    
    # Plotting the misclassification error vs depth of tree for both testing/training
    plt.plot(range(1,1+len(mis_train)), mis_train, linestyle='-', marker='o')
    plt.plot(range(1,1+len(mis_test)), mis_test, linestyle='-', marker='o')
    plt.legend(['Training Data', 'Testing Data'], loc='upper right')
    plt.ylabel("Misclassification error (%)")
    plt.xlabel("Depth of Tree")
    plt.title("Misclassification Errors Vs. Depth of Trees - " + str(data))
    plt.axis([0,depth,0,40])
    plt.show()
    
    # Printing out the minimum classification error for testing data set
    min = np.array([[pd.DataFrame.min(mis_test.Error), 
                     pd.Series.idxmin(mis_test)+1]])
    headers = ['Minimum Test Error', 'Tree Depth']
    table = tabulate(min, headers, tablefmt="fancy_grid")
    print(table)
    return dec_tree;

# Running the function on the MADELON data with max depth from 1-12 decision trees
start_time = time.time()
a = DECISION_TREES(train=X_train, trainlab=y_train,
             test=X_test,testlab=y_test,
             depth=12, data = "Autism" )
print("--- %s seconds ---" % (time.time() - start_time))

dec_tree=DecisionTreeClassifier(max_depth=5) # Max_Depth attribute
dec_tree=dec_tree.fit(X_train, y_train) # Growing out the decision tree

# Predicting on training data set / calculating misclassification error
train_pred=dec_tree.predict(X_train)
mis_train = (accuracy_score(y_train,train_pred) * 100)

# Predicting on testing data set / calculating misclassification error
test_pred=dec_tree.predict(X_test)
mis_test= (accuracy_score(y_test,test_pred)*100)

print(mis_train)
print(mis_test)
print("--- %s seconds ---" % (time.time() - start_time))

# CHOSE TREE DEPTH 6, SMALLEST ERROR BEFORE OVERFITTING


#Visualize Decision Tree

dot_data = StringIO()
export_graphviz(dec_tree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())    



### /////////////////////////////////////////////////////////////////////////
### /////////////////////////// Random Forests /////////////////////////////
### ///////////////////////////////////////////////////////////////////////

def RandomTreePlot(train, trainlab, valid, validlab, features):
    # Defining the number of trees (k)
    k= [1,3,5,8,10,20,30,40,50,60,70,80,90,100,300]
    
    # Initialization of misclassification error matrix
    d = np.zeros((15,3))
    d[:,0] = k
    Error_train = np.zeros(15)
    Error_test = np.zeros(15)
    
    # Creating random forests with k number of trees and the split attribute 
    # specified for TRAINING data
    for x in range(0,15):
        rand_forest=RandomForestClassifier(n_estimators=k[x], max_features=features)
        rand_forest = rand_forest.fit(train, trainlab.values.ravel())
        predictions = rand_forest.predict(train)
        Error_train[(x)] = 1 - metrics.accuracy_score(trainlab,predictions)
    
    # Creating random forests with k number of trees and the split attribute 
    # specified for TESTING data
    for x in range(0,15):
        rand_forest=RandomForestClassifier(n_estimators=k[x], max_features=features)
        rand_forest = rand_forest.fit(train, trainlab.values.ravel())
        predictions = rand_forest.predict(valid)
        Error_test[(x)] = 1 - metrics.accuracy_score(validlab, predictions)
    
    # Assignment of misclassification errors for training/testing data sets to 
    # misclassification error matrix
    d[:,1] = Error_train
    d[:,2] = Error_test  
    
    # Plotting the misclassification error vs number of trees for both 
    # testing/training
    plt.plot(d[:,0], d[:,1], linestyle='-', marker='o')
    plt.plot(d[:,0], d[:,2], linestyle='-', marker='o')
    plt.legend(['Training Data', 'Testing Data'], loc='upper right')
    plt.ylabel("Misclassification error")
    plt.xlabel("Number of Tree")
    plt.title("Misclassification Errors Vs. Number of Trees")
    plt.show()
    
    headers = ['Tree Depth', 'Misclassification error - Train', "Misclassification error - Test",]
    table = tabulate(d, headers, tablefmt="fancy_grid")
    print(table)
    return rand_forest
    
    
# The split attribute being chosen from a random subset of sqrt(500) features
start_time = time.time()
rand_forest = RandomTreePlot(train=X_train, trainlab=y_train,
               valid=X_test,validlab=y_test, features="sqrt")
print("--- %s seconds ---" % (time.time() - start_time))



### /////////////////////////////////////////////////////////////////////////
### ////////////////////////////// LogitBoost //////////////////////////////
### ///////////////////////////////////////////////////////////////////////

# Reading in labels as -1/1 instead of 0/1
autism_labels2 = pd.read_csv("/Users/SamCoplin/Downloads/AUTISMLABELS_FinalTAKE2.csv")

# Splitting data set into test and train data set  (80%/20%)
X_train, X_test, y_train, y_test = train_test_split(autism, autism_labels2, test_size=0.2, random_state=1234)

X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)

start_time = time.time()

# Adding a column of 1's to xtrain and xtest data sets
X_train = pd.concat([pd.DataFrame([1]*len(X_train)), X_train], axis=1)
X_test = pd.concat([pd.DataFrame([1]*len(X_test)), X_test], axis=1)

# Creating a label column heading
y_train.columns = ['label']


# Initializing N and M 
N = len(X_train)
M = X_train.shape[1]
  
# Iteration 
weightsall = np.transpose(np.zeros(M))
loss = np.empty([300, 1])
error_train = np.empty([4, 1]) # MAKE SURE THESE ARE ZERO BEFORE RUNNING LOOP
error_test = np.empty([4, 1]) # MAKE SURE THESE ARE ZERO BEFORE RUNNING LOOP
ik = 1
z = np.zeros(N) 
  

# Loop 
for i in range(1,301):
    H = np.dot(X_train, weightsall)
    p = (1/(1+np.exp(-2*H)))
    w = p*(1-p)
    y_train.reset_index(drop=True, inplace=True)
    for t in range(len(w)):
        if w[t] == 0:
            z[t] = 0
        else:
            z[t] = (((.5*(y_train.label[t]+1))-p[t])/w[t])
          
    coef = np.empty([2,M-1])    
    newloss = np.empty([M-1, 1])
      
    for j in range(0,M-1):
        Xj = X_train.values[:,j+1]   
        a = sum(w)
        b = sum(w*Xj)
        c = sum(w*Xj**2)
        d = sum(w*z)
        e = sum(w * Xj * z)
          
        # WLS estimation
        if (a*c)-b**2 ==0:
            Bj = np.array([(d/a), 0])
        else:
            Bj = ( 1 / (a*c-b**2) ) * np.array([c * d - b * e, a * e - b* d])
          
        # New H(X)
        Hj = H + .5*(Bj[0] + Bj[1] * Xj) 
          
        # New loss function value
        lossj = sum(np.log(1 + np.exp(-2*y_train.label * Hj)))
  
        coef[:, j]=Bj
        newloss[j]=lossj
          
    Jhat = np.argmin(newloss) # [1:len(newloss)]
          
    weightsall[0] = weightsall[0] + .5*coef[0,Jhat]
    weightsall[Jhat+1] = weightsall[Jhat+1]+.5*coef[1,Jhat]
    loss[i-1] = newloss[Jhat]
      
    # Making predictions
    pred_train = np.sign(np.dot(X_train,weightsall))
    pred_test = np.sign(np.dot(X_test,weightsall))
      
    # Calculating error rate
    if i in (10,30, 100, 300):
        error_train[ik-1]=1-metrics.accuracy_score(pred_train, y_train)
        error_test[ik-1]=1-metrics.accuracy_score(pred_test, y_test)
        ik = ik+1
      
    if i==300:
        plt.plot(range(1,301), loss)
        
print("--- %s seconds ---" % (time.time() - start_time))

X_train = X_train.drop(0,axis = 1)
X_test = X_test.drop(0,axis = 1)


### /////////////////////////////////////////////////////////////////////////
### //////////////////////////// NeuralNetworks ////////////////////////////
### ///////////////////////////////////////////////////////////////////////
def neuralnets(k, lr):
    accuracy_train = np.empty([10, 1]) # MAKE SURE THESE ARE ZERO BEFORE RUNNING LOOP
    accuracy_test = np.empty([10, 1])
    
        
    # Split training and testing data set 
    train_x, test_x, train_y, test_y = train_test_split(autism, autism_labels2, test_size=0.2)

    # Normalize data set 
    scaler = preprocessing.StandardScaler().fit(train_x)
    train_x = pd.DataFrame(scaler.transform(train_x))
    
    # Transform test data set                       
    test_x = pd.DataFrame(scaler.transform(test_x))
    
    # Creating a model
    model = Sequential()
    model.add(Dense(k, input_dim=train_x.shape[1], activation='relu'))
    model.add(Dense(k, input_dim=train_x.shape[1], activation='relu'))
    model.add(Dense(1))
    
    # Compiling model
    sgd = optimizers.SGD(lr=lr)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    
    # Training a model
    results = model.fit(train_x, train_y, validation_data = (test_x, test_y), epochs=100, batch_size=100)
    
    accuracy_train = results.history['acc'][99]
    accuracy_test = results.history['val_acc'][99]
    return accuracy_train, accuracy_test

start_time = time.time()
mini_k32_2layers = neuralnets(32, .01)
print("--- %s seconds ---" % (time.time() - start_time))

#89, 89

mini_k32_2layers = neuralnets(128, .01)
# 92, 91

mini_k32_2layers = neuralnets(128, .001)
# 79, 78

mini_k32_2layers



### /////////////////////////////////////////////////////////////////////////
### ///////////////////////// Question Importance //////////////////////////
### ///////////////////////////////////////////// //////////////////////////
### ///////////////////////////////////////////////////////////////////////

oddsratios = np.vstack((list(X_train),np.exp(logreg.coef_)))
QuestionImportance = np.vstack((oddsratios,dec_tree.feature_importances_))
QuestionImportance = np.vstack((QuestionImportance,rand_forest.feature_importances_))
QuestionImportance = pd.DataFrame(QuestionImportance)
QuestionImportance = QuestionImportance.drop([10,11,12,13,14,15,16,17,18,19,20,21], axis=1)
QuestionImportance = QuestionImportance.T
QuestionImportance.columns = ['Variable','OddsRatio','Decision Tree Feature Importance','Random Forest Feature Importance']