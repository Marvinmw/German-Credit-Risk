from numpy.random import seed
import lenet
import keras
import dataprovider as dp
seed(34)
#Using LeNet.
# Removing the max pooling layer.
# Nestorv optimization
(xtrain,ytrain),(xtext,ytest) = dp.loaddata()

ytrain = keras.utils.to_categorical(ytrain, 2)
ytest = keras.utils.to_categorical(ytest, 2)
cnnlenet = lenet.letnet4((9,),2)
lenet.train(cnnlenet,xtrain, ytrain, xtext, ytest, lr=0.01, epoches=200,
                     batch_size=800, modelpath='./lenet.h5', historypath='./lenet_history.csv',
                     nesterov=True)


