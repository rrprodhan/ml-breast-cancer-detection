import pandas as pd
import numpy as np
import csv
from urllib.request import urlopen

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

with open("breast-cancer.data.csv", "w") as refactored_file:

    theWriter = csv.writer(refactored_file)
    theWriter.writerow(['class','age','menopause','tumor-size','inv-nodes','node-caps','deg-malig','breast','breast-quad','irradiat'])

    for line in urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data"):

        #Cleaning up the whole dataset
        refactoredLine = line.decode('UTF-8')
        refactoredLine = refactoredLine.replace('?', '')
        refactoredLine = refactoredLine.replace('"', '')
        refactoredLine = refactoredLine.replace('\'', '')
        refactoredLine = refactoredLine.replace('_', '')
        refactoredLine = refactoredLine.lower()

        #Making all categorical type datas numeric
        #1.Class: no-recurrence-events = 0 & recurrence-events = 1
        refactoredLine = refactoredLine.replace('no-recurrence-events', '0')
        refactoredLine = refactoredLine.replace('recurrence-events', '1')
        #2.Age: 10-19=0,20-29=1,30-39=2,40-49=3,50-59=4,60-69=5,70-79=6,80-89=7,90-99=8
        refactoredLine = refactoredLine.replace('10-19', '0')
        refactoredLine = refactoredLine.replace('20-29', '1')
        refactoredLine = refactoredLine.replace('30-39', '2')
        refactoredLine = refactoredLine.replace('40-49', '3')
        refactoredLine = refactoredLine.replace('50-59', '4')
        refactoredLine = refactoredLine.replace('60-69', '5')
        refactoredLine = refactoredLine.replace('70-79', '6')
        refactoredLine = refactoredLine.replace('80-89', '7')
        refactoredLine = refactoredLine.replace('90-99', '8')
        #3.Menopause: lt40=0,ge40=1
        refactoredLine = refactoredLine.replace('lt40', '0')
        refactoredLine = refactoredLine.replace('ge40', '1')
        refactoredLine = refactoredLine.replace('premeno', '2')
        #4.Tumor-size: 0-4=0,5-9=1,10-14=2,15-19=3,20-24=4,25-29=5,
        #30-34=6,35-39=7,40-44=8,45-49=9,50-54=10,55-59=11
        refactoredLine = refactoredLine.replace('0-4', '0')
        refactoredLine = refactoredLine.replace('5-9', '1')
        refactoredLine = refactoredLine.replace('10-14', '2')
        refactoredLine = refactoredLine.replace('15-19', '3')
        refactoredLine = refactoredLine.replace('20-24', '4')
        refactoredLine = refactoredLine.replace('25-29', '5')
        refactoredLine = refactoredLine.replace('30-34', '6')
        refactoredLine = refactoredLine.replace('35-39', '7')
        refactoredLine = refactoredLine.replace('40-44', '404')
        refactoredLine = refactoredLine.replace('45-49', '9')
        refactoredLine = refactoredLine.replace('50-54', '10')
        refactoredLine = refactoredLine.replace('55-59', '11')
        #5.Inv-nodes: 0-2=0,3-5=1,6-8=2,9-11=3,12-14=4,15-17=5,18-20=6,21-23=7
        #24-26=8,27-29=9,30-32=10,33-35=11,36-39=12
        refactoredLine = refactoredLine.replace('0-2', '0')
        refactoredLine = refactoredLine.replace('3-5', '1')
        refactoredLine = refactoredLine.replace('6-8', '2')
        refactoredLine = refactoredLine.replace('9-11', '3')
        refactoredLine = refactoredLine.replace('12-14', '4')
        refactoredLine = refactoredLine.replace('15-17', '5')
        refactoredLine = refactoredLine.replace('18-20', '6')
        refactoredLine = refactoredLine.replace('21-23', '7')
        refactoredLine = refactoredLine.replace('24-26', '8')
        refactoredLine = refactoredLine.replace('27-29', '9')
        refactoredLine = refactoredLine.replace('30-32', '10')
        refactoredLine = refactoredLine.replace('33-35', '11')
        refactoredLine = refactoredLine.replace('36-39', '12')
        #6.Node-caps & 10.Irradiat: yes=1 & no=0
        refactoredLine = refactoredLine.replace('yes', '1')
        refactoredLine = refactoredLine.replace('no', '0')
        #9.Breast-quad: leftup=0,leftlow=1,rightup=2,rightlow=3,central=4
        refactoredLine = refactoredLine.replace('leftup', '0')
        refactoredLine = refactoredLine.replace('leftlow', '1')
        refactoredLine = refactoredLine.replace('rightup', '2')
        refactoredLine = refactoredLine.replace('rightlow', '3')
        refactoredLine = refactoredLine.replace('central', '4')
        #8.Breast: left=0 & right=1
        refactoredLine = refactoredLine.replace('left', '0')
        refactoredLine = refactoredLine.replace('right', '1')

        refactored_file.write(refactoredLine.strip() + '\n')
    refactored_file.close()

breastCancerDataset = pd.read_csv("breast-cancer.data.csv")
breastCancerDataset.fillna(breastCancerDataset.mean(), inplace=True) #Filling up all blank cells with mean of the particular column

rfc = RandomForestClassifier(n_estimators=2000) #Will generate 2000 Trees for clssification
train, test = train_test_split(breastCancerDataset, test_size=0.2) #80% train and 20% test data split
train_features = train.iloc[:,1:10]
train_target = train["class"]
test_features = test.iloc[:,1:10]
test_target = test["class"]
rfc.fit(train_features,train_target)

predicted_target = rfc.predict(test_features)

print("Confusion matrix: \n")
print(confusion_matrix(test_target,predicted_target))
print("\nClassification report: \n")
print(classification_report(test_target,predicted_target))
print("Accuracy: {}".format(accuracy_score(test_target, predicted_target)))
