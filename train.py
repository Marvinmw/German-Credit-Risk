import model as m
import dataprovider as dp
from numpy.random import seed
seed(34)

(xtrain,ytrain),(xtext,ytest) = dp.loaddata()

# SVM model
print('SVM classifier')
svmmodel = m.svm((xtrain,ytrain),(xtext,ytest))

# Decision Tree
print('Decision Tree')
dtmodel = m.decisionTree((xtrain,ytrain),(xtext,ytest))

# Random Tree
print('Random Tree')
rtmodel =  m.randomTree((xtrain,ytrain),(xtext,ytest))

#Logistic Classifier
# Random Tree
print('Logistic classifier')
logistimodel = m.logisticClassifer((xtrain,ytrain),(xtext,ytest))

#knn
print('KNN')
knn = m.knn((xtrain,ytrain),(xtext,ytest))

#adaboosting
print('Adaboosting')
adaboosting = m.adaboosting((xtrain,ytrain),(xtext,ytest))

#voting
print('Voting')
classifiers = [('svm',svmmodel),('dt',dtmodel),('rt',rtmodel),('lt',logistimodel),('knn',knn),
               ('addboosting',adaboosting)]
votingmodel = m.voting((xtrain,ytrain),(xtext,ytest),classifiers)




