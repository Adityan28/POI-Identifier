#!/usr/bin/python

import sys
import pickle
import math
import matplotlib.pyplot as plt
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn import preprocessing

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
##features_list = ['poi','salary','deferral_payments', 'total_payments', 'loan_advances', 'bonus',
##                 'restricted_stock_deferred','deferred_income', 'total_stock_value', 'expenses',
##                 'exercised_stock_options', 'other', 'long_term_incentive','restricted_stock',
##                 'director_fees', 'shared_receipt_with_poi','to_messages','from_messages'] # You will need to use more features

features_list = ['poi','salary', 'total_payments', 'exercised_stock_options', 'deferred_income','from_messages', 'to_messages','from_poi_to_this_person','from_this_person_to_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
##print "LENLEN: ", len(data_dict)
### Task 2: Remove outliers
count = 0    
for d in data_dict:
    if data_dict[d]['poi'] == 1:
        count += 1
print "POIs: ", count
print "Features: ", len(data_dict[d])
data_dict.pop('TOTAL',0)
data_dict.pop('SAVAGE FRANK',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)

##keys = data_dict.keys()

for f in features_list:
    all_ppl = [data_dict[d][f] for d in data_dict]
    for i in range(len(all_ppl)):
        if all_ppl[i] == 'NaN':
            all_ppl[i]=0
    i=0
    for d in data_dict:
        data_dict[d][f] = all_ppl[i]
        i += 1

features_list.remove('from_messages')
features_list.remove('to_messages')
features_list.remove('from_poi_to_this_person')
features_list.remove('from_this_person_to_poi')
features_list.remove('deferred_income')
##features_list.remove('total_payments')
features_list.append('percentage_from_this_person_to_poi')
features_list.append('percentage_from_poi_to_this_person')
features_list.append('deferred_income_to_total_payments')

### Task 3: Create new feature(s)
for d in data_dict:
    if data_dict[d]['total_payments']>0:
        data_dict[d]['deferred_income_to_total_payments']=float(data_dict[d]['deferred_income'])/float(data_dict[d]['total_payments'])
    else:
        data_dict[d]['deferred_income_to_total_payments']=0

    if data_dict[d]['to_messages']!=0:
        data_dict[d]['percentage_from_this_person_to_poi']=float(data_dict[d]['from_this_person_to_poi'])/float(data_dict[d]['to_messages'])
    else:
        data_dict[d]['percentage_from_this_person_to_poi']=0
    if data_dict[d]['from_messages']!=0:
        data_dict[d]['percentage_from_poi_to_this_person']=float(data_dict[d]['from_poi_to_this_person'])/float(data_dict[d]['from_messages'])
    else:
        data_dict[d]['percentage_from_poi_to_this_person']=0

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)

labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.metrics import accuracy_score,precision_score,recall_score

from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size = .3, random_state = 42)

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import GridSearchCV

##print "\n*********GaussianNB()**********\n"
##from sklearn.naive_bayes import GaussianNB
##clf = GaussianNB()
##clf.fit(features_train, labels_train)
##pred = clf.predict(features_test)
##accuracy = accuracy_score(pred, labels_test)
##print "Accuracy of prediction: ", accuracy
##precision = precision_score(labels_test,pred)
##print "Precision of prediction: ", precision
##recall = recall_score(labels_test, pred)
##print "Recall of prediction: ", recall, "\n"
##
##print "\n*********SVM**********\n"
##from sklearn import svm
##clf = svm.SVC(kernel='rbf', C=10)
##clf.fit(features_train, labels_train)
##pred = clf.predict(features_test)
##accuracy = accuracy_score(pred, labels_test)
##print "Accuracy of prediction: ", accuracy
##precision = precision_score(labels_test,pred)
##print "Precision of prediction: ", precision
##recall = recall_score(labels_test, pred)
##print "Recall of prediction: ", recall, "\n"

##print "\n*********Decision Tree Classifier**********\n"
##from sklearn import tree
####params = {'min_samples_split':[2,3,4,5,6,7,8,9,10], 'criterion':('gini','entropy'), 'max_depth':[None,2,5,10,20,50,100]}
####dtc = tree.DecisionTreeClassifier()
####clf = GridSearchCV(dtc, params)
##clf = tree.DecisionTreeClassifier(min_samples_split=5, criterion='gini', max_depth=20)
##clf.fit(features_train, labels_train)
##pred = clf.predict(features_test)
####bestParams = clf.best_params_
####print "My param fams: ", bestParams
##accuracy = accuracy_score(pred, labels_test)
##print "Accuracy of prediction: ", accuracy
##precision = precision_score(labels_test,pred)
##print "Precision of prediction: ", precision
##recall = recall_score(labels_test, pred)
##print "Recall of prediction: ", recall, "\n"

##print "\n*********AdaBoost Classifier**********\n"
from sklearn.ensemble import AdaBoostClassifier
##adaParams = {'n_estimators':[1,100], 'random_state':[1,10]}
##aBc = AdaBoostClassifier()
##clf = GridSearchCV(aBc, adaParams)
clf = AdaBoostClassifier(n_estimators=100, random_state=1)
clf.fit(features_train, labels_train)
predADB = clf.predict(features_test)
##bestParams = clf.best_params_
feat_imp = clf.feature_importances_
print "Feature Importances: ", feat_imp
##accuracy = accuracy_score(predADB, labels_test)
##print "Accuracy of prediction: ", accuracy
##precision = precision_score(labels_test,predADB)
##print "Precision of prediction: ", precision
##recall = recall_score(labels_test, predADB)
##print "Recall of prediction: ", recall, "\n"
print "Great. Now run the tester.py script to evaluate your trained classifier!"

##print "\n*********Random Forest Classifier**********\n"
##from sklearn.ensemble import RandomForestClassifier
####rndParams = {'n_estimators':[1,100], 'random_state':[1,10]}
####rFc = RandomForestClassifier()
####clf = GridSearchCV(rFc, rndParams)
##clf = RandomForestClassifier(n_estimators=100, random_state=1)
##clf.fit(features_train, labels_train)
##predRND = clf.predict(features_test)
##bestParams = clf.best_params_
##accuracy = accuracy_score(predRND, labels_test)
##print "Accuracy of prediction: ", accuracy
##precision = precision_score(labels_test,predRND)
##print "Precision of prediction: ", precision
##recall = recall_score(labels_test, predRND)
##print "Recall of prediction: ", recall, "\n"

##print "\n*********K Neighbors Classifier**********\n"
##from sklearn.neighbors import KNeighborsClassifier
####knParams = {'n_neighbors':[1,30], 'weights':('uniform','distance'), 'algorithm' : ('auto','ball_tree','kd_tree','brute')}
####kNc = KNeighborsClassifier()
####clf = GridSearchCV(kNc, knParams)
##clf = KNeighborsClassifier(n_neighbors=2, weights='uniform')
##clf.fit(features_train, labels_train)
##predKN = clf.predict(features_test)
####bestParams = clf.best_params_
##accuracy = accuracy_score(predKN, labels_test)
##print "Accuracy of prediction: ", accuracy
##precision = precision_score(labels_test,predKN)
##print "Precision of prediction: ", precision
##recall = recall_score(labels_test, predKN)
##print "Recall of prediction: ", recall, "\n"


# Fit and Predict with test data after removal of outliers

##print "Total no. now:", len(cleanedData)
##f_train, l_train = zip(*cleanedData)
##clf.fit(f_train, l_train)
##pred = clf.predict(features_test)
##accuracy = accuracy_score(pred, labels_test)
##print "Accuracy of prediction: ", accuracy

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
##from sklearn.cross_validation import train_test_split
##features_train, features_test, labels_train, labels_test = \
##    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
