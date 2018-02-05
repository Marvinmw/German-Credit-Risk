from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn import linear_model
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier

def svm(train_data,test_data,kernel='rbf',C=1):
    model = SVC(kernel=kernel,C=C)
    scores = cross_val_score(model,train_data[0],train_data[1],cv=5)
    print("Train Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    #predictedy = model.predict(test_data[0])
    model.fit(train_data[0],train_data[1])
    testscore = model.score(test_data[0],test_data[1])
    print("Test Accuracy: %0.2f " % (testscore.mean()))
    return model


def decisionTree(train_data,test_data):
    model = tree.DecisionTreeClassifier()
    scores = cross_val_score(model, train_data[0], train_data[1], cv=5)
    print("Train Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # predictedy = model.predict(test_data[0])
    model.fit(train_data[0], train_data[1])
    testscore = model.score(test_data[0], test_data[1])
    print("Test Accuracy: %0.2f " % (testscore.mean()))
    return model


def randomTree(train_data,test_data,state=23):
    model = RandomForestClassifier(random_state=state)
    scores = cross_val_score(model, train_data[0], train_data[1], cv=5)
    print("Train Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # predictedy = model.predict(test_data[0])
    model.fit(train_data[0], train_data[1])
    testscore = model.score(test_data[0], test_data[1])
    print("Test Accuracy: %0.2f " % (testscore.mean()))
    return model

def logisticClassifer(train_data,test_data):
    model =  linear_model.LogisticRegression()
    scores = cross_val_score(model, train_data[0], train_data[1], cv=5)
    print("Train Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # predictedy = model.predict(test_data[0])
    model.fit(train_data[0], train_data[1])
    testscore = model.score(test_data[0], test_data[1])
    print("Test Accuracy: %0.2f " % (testscore.mean()))
    return model

def knn(train_data,test_data, n_neighbors=15):
    model = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
    #scores = cross_val_score(model, train_data[0], train_data[1], cv=5)
    #print("Train Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(model, train_data[0], train_data[1], cv=5)
    print("Train Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    model.fit(train_data[0], train_data[1])
    testscore = model.score(test_data[0], test_data[1])
    print("Test Accuracy: %0.2f " % (testscore))
    return model

def adaboosting(train_data,test_data,base_model=tree.DecisionTreeClassifier()):
    model = AdaBoostClassifier(base_model,
                             algorithm="SAMME",
                             n_estimators=200)
    scores = cross_val_score(model, train_data[0], train_data[1], cv=5)
    print("Train Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    model.fit(train_data[0], train_data[1])
    testscore = model.score(test_data[0], test_data[1])
    print("Test Accuracy: %0.2f " % (testscore))
    return model


def voting(train_data,test_data,classifiers):
    model = VotingClassifier(estimators=classifiers)
    scores = cross_val_score(model, train_data[0], train_data[1], cv=5)
    print("Train Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    model.fit(train_data[0], train_data[1])
    testscore = model.score(test_data[0], test_data[1])
    print("Test Accuracy: %0.2f " % (testscore))
    return model
