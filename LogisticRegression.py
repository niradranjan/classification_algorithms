import sys
import csv
import math

# converts the csv file into a list containing rows of data
# returns the dataset
def fileLoader(file):
    dataset = list()
    with open(file) as dataFile:
        rows = csv.reader(dataFile)
        for row in rows:
            dataset.append(row)
    return dataset

# convert all the nominal values in the
# dataset to numerical values
def convertToFloat(dataset, i):
    for row in dataset:
        row[i] = float(row[i].strip())

# find the minimum and maximum value of each column and store the pair in a list
# the list contains 9 pairs of numbers corresponding to the 9 columns
# in the dataset
def findMinMax(dataset):
    keyValList = list()
    for i in range(len(dataset[0])):
        val = [row[i] for row in dataset]
        minimum = min(val)
        maximum = max(val)
        keyValList.append([minimum, maximum])
    return keyValList

# the minimum and maximum value already found out are used in this method
# to normalize all the values in the dataset
# the range of values will lie between 0 and 1
def normalize(dataset, keyValList):
    for row in dataset:
        for i in range(len(row)):
            num = row[i] - keyValList[i][0]
            den = keyValList[i][1] - keyValList[i][0]
            row[i] = num/den


# splits the given dataset into training set and test set according
# to the given split ratio.Returns both the sets
def split(dataSet,splitSize):
    trainSet = []
    testSet = []
    rowsize = len(dataSet[0])
    for x in range(0, len(dataSet)):
        if x < math.ceil(len(dataSet) * splitSize):
            trainSet.append(dataSet[x])
        else:
            testSet.append(dataSet[x])
    return trainSet, testSet


#finds the accuracy of the predictions.
# finds the count of true positives,false positives,true negatives, false negatives
# which hep in finding the confusion matrix
def accuracy_metric(testSet, predicted):
    correct = 0
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    actual = [row[-1] for row in testSet]
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
        if actual[i] == 1 and predicted[i] == 1:
            tp += 1
        if actual[i] == 0 and predicted[i] == 0:
            tn += 1
        if actual[i] == 1 and predicted[i] == 0:
            fn += 1
        if actual[i] == 0 and predicted[i] == 1:
            fp += 1

    accuracy = correct / float(len(actual)) * 100.0
    print("Accuracy : " + `accuracy`)
    print("True Positive : " + `tp`)
    print("True Negative : " + `tn`)
    print("False Negative : " + `fn`)
    print("False Positive : " + `fp`)

# normal implementation of the logistic function.This method can
# be used without the stochastic gradient descent method to make
# predictions as well.It takes the attributes and their related
# weights and finds the new y value
def predict(row, weights):
    hyp = weights[0]
    for i in range(len(row) - 1):
        hyp += weights[i + 1] * row[i]
    return 1.0 / (1.0 + math.exp(-hyp))

# stochastic gradient descent logic implemented in this method.For each iteration, a prediction
# is made using the logistic funcation.This predicted value is then compared
# with the actual class value to find the error.Then the weights are updated so that in the
# next iteration, the amount of error is reduced.In each iteration all the weights for all the
# attributes are updated.The learning rate provided is used in the weight update equation
def sgd(train, learningRate, noOfIterations):
    weights = [0.0 for i in range(len(train[0]))]
    for epoch in range(noOfIterations):
        for row in train:
            hyp = predict(row, weights)
            error = row[-1] - hyp
            weights[0] = weights[0] + learningRate * error * hyp * (1.0 - hyp)
            for i in range(len(row) - 1):
                weights[i + 1] = weights[i + 1] + learningRate * error * hyp * (1.0 - hyp) * row[i]
    return weights

# logistic regression implementation.The final optimal set of weights
# obtained after using stochastic gradient descent is used to make predictions
# for the class value of the test sets instances.The final values calculated are rounded off
# to either zero or one for ease of predictions in the logistic regression classifier
def logReg(train, test, learningRate, noOfIterations):
    predictions = list()
    coef = sgd(train, learningRate, noOfIterations)
    for row in test:
        hyp = predict(row, coef)
        hyp = round(hyp)
        predictions.append(hyp)
    return (predictions)



#starting method of the class
#all function calls are made from this method
def starterMethod(file,learningRate,noOfIterations):
    filePath = file
    learningRate = float(learningRate)
    noOfIterations = int(noOfIterations)
    dataset = fileLoader(filePath)
    for i in range(len(dataset[0])):
        convertToFloat(dataset, i)
    keyValList = findMinMax(dataset)
    normalize(dataset, keyValList)
    trainSet, testSet = split(dataset, 0.7)
    predicted = logReg(trainSet, testSet, learningRate, noOfIterations)
    accuracy_metric(testSet, predicted)

#main method
if __name__ == "__main__":
    file = sys.argv[1]
    learningRate = sys.argv[2]
    noOfIterations = sys.argv[3]
    starterMethod(file,learningRate,noOfIterations)