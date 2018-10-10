import load_dataset
import numpy as np
import heapq
from collections import Counter
import time
import random

startTime = time.time()
class ImageData:
    def __init__(self, image, label):
        self.image = image
        self.label = label

class BallTreeNode:
    def __init__(self, image):
        self.imageData = image
        self.left = None
        self.right = None

trainingData = load_dataset.read("training")
testData = load_dataset.read("testing")

size = 1

trainLbls = trainingData[0][:size * 6]
trainImgs = trainingData[1][:size * 6]
testLbls = testData[0][:size]
testImgs = testData[1][:size]
ks = [1, 3, 5, 10, 30, 50, 70, 80, 90, 100]

trainingData = []
testData = []

for i in range(len(trainLbls)):
    trainingData.append(ImageData(trainImgs[i], trainLbls[i]))

for i in range(len(testLbls)):
    testData.append(ImageData(testImgs[i], testLbls[i]))

def getCentroid(data):
    sums = [[0] * 28 for _ in range(28)]
    n = len(data)
    for d in data:
        for i in range(28):
            for j in range(28):
                sums[i][j] += d.image[i][j]
    
    for i in range(28):
        for j in range(28):
            sums[i][j] /= 784
    return sums

def getFurthest(target, data):
    furthest = (0, -1)
    for d in range(len(data)):
        distance = 0
        image = data[d].image
        for i in range(28):
            for j in range(28):
                distance += (image[i][j] - target[i][j]) ** 2
        if distance > furthest[0]:
            furthest = (distance, d)
    return data[furthest[1]]

def seperateTwoBalls(data, f1, f2):
    balls = [[], []]
    for d in range(len(data)):
        distance1 = 0
        distance2 = 0
        image = data[d].image
        for i in range(28):
            for j in range(28):
                distance1 += (image[i][j] - f1.image[i][j]) ** 2
                distance2 += (image[i][j] - f2.image[i][j]) ** 2
        if distance1 < distance2:
            balls[0].append(data[d])
        else:
            balls[1].append(data[d])
    return balls

def constructBallTree(training):
    if len(training) == 1:
        return BallTreeNode(training[0])
    else:
        centroid = ImageData(getCentroid(training), -1)
        ballNode = BallTreeNode(centroid)
        
        f1 = getFurthest(centroid.image, training)
        f2 = getFurthest(f1.image, training)
        balls = seperateTwoBalls(training, f1, f2)
        
        ballNode.left = constructBallTree(balls[0])
        ballNode.right = constructBallTree(balls[1])

        return ballNode

root = constructBallTree(trainingData)

print("--- %s seconds ---" % (time.time() - startTime))


# def getPrediction(knn):
#     guess = [n[1] for n in knn]
#     return Counter(guess).most_common(1)[0][0]


# print("--- %s seconds ---" % (time.time() - startTime))

# for k in ks:
#     corrects = 0
#     for i in range(size):
#         testLbl = testLbls[i]
#         heap = []
#         knn = heapq.nsmallest(k, heap)
#         if getPrediction(knn) == testLbl:
#             corrects += 1
#     print(corrects / size)
    
# print("--- %s seconds ---" % (time.time() - startTime))

