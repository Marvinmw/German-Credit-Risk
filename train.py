import model as m
import dataprovider as dp
from numpy.random import seed
# import lenet
seed(34)

# SVM model
(xtrain,ytrain),(xtext,ytest) = dp.loaddata()
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


#Using LeNet.
# Removing the max pooling layer.
# Nestorv optimization
# import keras
# ytrain = keras.utils.to_categorical(ytrain, 2)
# ytest = keras.utils.to_categorical(ytest, 2)
# cnnlenet = lenet.letnet4((9,),2)
# lenet.train(cnnlenet,xtrain, ytrain, xtext, ytest, lr=0.01, epoches=200,
#                      batch_size=800, modelpath='./lenet.h5', historypath='./lenet_history.csv',
#                      nesterov=True)




