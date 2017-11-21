import sys
import csv
import math

# loads the csv file into a list.
# converts all the nominal values in the dataset into numerical values.
# Splits the input dataset into training set and test set according to the
# split size provided
def fileLoader(file,splitSize):
    trainSet = []
    testSet = []
    with open(file) as dataFile:
        rows = csv.reader(dataFile)
        dataSet = list(rows)
        rowsize = len(dataSet[0])
        for i in range(0,len(dataSet)):
            for j in range(0, rowsize -1):
                dataSet[i][j] = float(dataSet[i][j])
            if i < math.ceil(len(dataSet) * splitSize):
                trainSet.append(dataSet[i])
            else :
                testSet.append(dataSet[i])
    return trainSet,testSet

#all the nearestneighours of the test instance are passed to this function.
# polling is done among the neighbors and the class value present
# in the majority of the neighbours is assigned to the test instance
def classification(nearestNeighbours):
    ind = len(nearestNeighbours[0]) - 1
    classValueTrack = {}
    length = len(nearestNeighbours)
    for x in range(0,length):
        classValue = nearestNeighbours[x][ind]
        if classValue in classValueTrack:
            classValueTrack[classValue] += 1
        else:
            classValueTrack[classValue] = 1
    sortedClassValue = sorted(classValueTrack.iteritems(), key=lambda cvt: cvt[1])
    return sortedClassValue[-1][0]

#method is used to find out the euclidian distance between any
# two instances.The square root of the sum of the square of the difference
# of respective attributes of 2 instances gives the eucidian distance which is used
# as the measure to find distance between 2 points
def distaneBetweenPoints(instance1, instance2, length):
    distance = 0
    for x in range(0, length - 1):
        distance = distance + pow((instance1[x] - instance2[x]),2)
    return math.sqrt(distance)

#the distance of all the training instances from the test instance
# is found out.The training instances are arranged in the increasing order of
#the distance from the test instance.The top K values are picked up according to the'
#value passed by the user.These selected instances are the nearest neighbours of the test instance
def findNearestNeighbours(trainSet, testInstance, kVal ):
    appendedSet = []
    nearestNeighbours = []
    length = len(trainSet[0])
    for x in range(0, len(trainSet)):
        distance = distaneBetweenPoints(trainSet[x], testInstance, length)
        appendedSet.append((trainSet[x],distance))
    appendedSet.sort(key=lambda modSet: modSet[1])
    for x in range(0,kVal):
        nearestNeighbours.append(appendedSet[x][0])
    return nearestNeighbours

#method is used to find the accuracy of the algorithm and to find out
# the instances which were correctly predicted.The class values of the
#correctly predicted instances is given as output
def accuracyMetric(testSet, predictedValueSet):
    count = 0
    predictionDict = {}
    for x in range(len(testSet)):
        if testSet[x][-1] == predictedValueSet[x]:
            count += 1
            if predictedValueSet[x] in predictionDict:
                predictionDict[predictedValueSet[x]] += 1
            else:
                predictionDict[predictedValueSet[x]] = 1
    accuracy = (count/float(len(testSet))) * 100.0
    print("Accuracy is : " + `accuracy`)
    print
    print("Correct Predictions")
    for key, value in predictionDict.iteritems():
        print(`key` +" : "+ `value`)

#starting method of the class
#all function calls are made from this method
def starterMethod(file,kValue):
    splitSize = 0.7
    kVal = int(kValue)
    trainSet, testSet = fileLoader(file, splitSize)
    predictedValueSet = []
    for x in range(len(testSet)):
        nearestNeighbours = findNearestNeighbours(trainSet, testSet[x], kVal)
        predictedValue = classification(nearestNeighbours)
        predictedValueSet.append(predictedValue)
    accuracyMetric(testSet, predictedValueSet)

#main method
if __name__ == "__main__":
    file = sys.argv[1]
    kValue = sys.argv[2]
    starterMethod(file,kValue)