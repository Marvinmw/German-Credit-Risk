from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense,Reshape
from keras.layers import Conv2D as Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
import  pandas
import  keras
from keras.callbacks import ModelCheckpoint

def letnet4(input_shape, classes):
    # initialize the model
    model = Sequential()
    wini = 'he_uniform'
    # first set of CONV => RELU => POOL
    #model.add(Dense(32,input_shape=input_shape,name='dense_in'))
    model.add(Reshape((3, 3,1), input_shape=input_shape))
    model.add(Convolution2D(20, (5, 5), padding="same",kernel_initializer=wini, name='conv1'))
    model.add(Activation("relu"))


    # second set of CONV => RELU => POOL
    model.add(Convolution2D(50, (5, 5), padding="same",kernel_initializer=wini,name='l2'))
    model.add(Activation("relu"))

    # set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500, name='dense1',kernel_initializer=wini))
    model.add(Activation("relu"))

    # softmax classifier
    model.add(Dense(classes,name='classifer'))
    model.add(Activation("softmax"))

    # return the constructed network architecture
    return model

def train(model, x_train, y_train, x_test, y_test, lr=0.01, epoches=12,
                     batch_size=200, modelpath='', historypath='',
                     nesterov=True):
        sgd = keras.optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=nesterov)
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=sgd, metrics=['accuracy'])

        checkpoint = ModelCheckpoint(modelpath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        # losstfboard = TensorBoardBatch(log_dir=tensorbatch, batch_size=batch_size)
        #tensorboard = TensorBoard(log_dir=tensorepoch)
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epoches,
                            verbose=1,
                            validation_data=(x_test, y_test),
                            callbacks=[checkpoint])
        pandas.DataFrame(history.history).to_csv(historypath)
        # model.save(modelpath, overwrite=True)
        return model, history