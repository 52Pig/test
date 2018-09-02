
from bayes_text_classify import bayes_lib



def loadDataSet():
    postingList = [
        ['my', 'dog', 'has'],
        ['maybe', 'not'],
        ['my', 'dalmation'],
        ['stop', 'posting'],
        ['mr', 'licks', 'ate', 'my', 'steak'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec







dataSet, listClasses = loadDataSet()
nb = bayes_lib.NBayes()
nb.train_set(dataSet, listClasses)
nb.map2vocab(dataSet[0])
print(nb.predict(nb.testset))