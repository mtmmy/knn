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
    def __init__(self, image, radius, leafs):
        self.imageData = image
        self.radius = radius
        self.left = None
        self.right = None
        self.leafs = leafs

trainingData = load_dataset.read("training")
testData = load_dataset.read("testing")

size = 400

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

def getDistance(image1, image2):
    sqrtSum = 0
    for i in range(28):
        for j in range(28):
            sqrtSum += (int(image1[i][j]) - image2[i][j]) ** 2
    return sqrtSum

def getCentroid(allPoints):
    sums = [[0] * 28 for _ in range(28)]
    for d in allPoints:
        for i in range(28):
            for j in range(28):
                sums[i][j] += d.image[i][j]
    
    for i in range(28):
        for j in range(28):
            sums[i][j] /= len(allPoints)
    return sums

def getFurthest(target, allPoints):
    furthest = (0, -1)
    for d in range(len(allPoints)):
        image = allPoints[d].image
        distance = getDistance(image, target)
        if distance > furthest[0]:
            furthest = (distance, d)
    return allPoints[furthest[1]]

def seperateTwoBalls(allPoints, f1, f2):
    balls = [[], []]
    for d in range(len(allPoints)):
        image = allPoints[d].image
        distance1 = getDistance(image, f1.image)
        distance2 = getDistance(image, f2.image)
        if distance1 < distance2:
            balls[0].append(allPoints[d])
        else:
            balls[1].append(allPoints[d])
    return balls

def constructBallTree(training):
    if len(training) == 1:
        return BallTreeNode(training[0], 0, 0)
    else:
        n = len(training)
        startPoint = random.choice(training)
        
        f1 = getFurthest(startPoint.image, training)
        f2 = getFurthest(f1.image, training)
        centroid = ImageData(getCentroid([f1, f2]), -1)
        balls = seperateTwoBalls(training, f1, f2)
        # print("ball 1: {}, ball 2: {}".format(len(balls[0]), len(balls[1])))

        radius = getDistance(f1.image, centroid.image)
        ballNode = BallTreeNode(centroid, radius, n)
        
        ballNode.left = constructBallTree(balls[0])
        ballNode.right = constructBallTree(balls[1])

        return ballNode

def searchBallTree(target, k, heap, ballTree, distance=-1):
    d = getDistance(target.image, ballTree.imageData.image) if distance == -1 else distance
    if len(heap) > k:
        curMax = heapq.heappop(heap)
        heapq.heappush(heap, curMax)
        if d - ballTree.radius >= -curMax[0]:
            return    
    if ballTree.leafs == 0:
        lbl = ballTree.imageData.label
        heapq.heappush(heap, (-d, lbl))
        if len(heap) > k:
            heapq.heappop(heap)
    else:
        d1 = getDistance(target.image, ballTree.left.imageData.image)
        d2 = getDistance(target.image, ballTree.right.imageData.image)
        if d1 < d2:
            searchBallTree(target, k, heap, ballTree.left, d1)
            searchBallTree(target, k, heap, ballTree.right, d2)
        else:
            searchBallTree(target, k, heap, ballTree.right, d2)
            searchBallTree(target, k, heap, ballTree.left, d1)
 
def getPrediction(knn):
    guess = [n[1] for n in knn]
    return Counter(guess).most_common(1)[0][0]

root = constructBallTree(trainingData)
constructTime = time.time()
print("--- Construct Ball Tree with Size {} spent {} seconds ---".format(size * 6, (constructTime - startTime)))

def test():
    correctness = {key: 0 for key in ks}
    for i in range(size):
        print("--- Testing the {}th data ---".format(i))
        testLbl = testLbls[i]
        knn = []
        k = 100
        searchBallTree(testData[i], k, knn, root)
        for k in ks:
            copyKnn = [(-d, l) for d, l in knn]
            heapq.heapify(copyKnn)
            times = k
            neighbors = []
            while times > 0 and copyKnn:
                neighbors.append(heapq.heappop(copyKnn))
                times -= 1
            if getPrediction(neighbors) == testLbl:
                correctness[k] += 1

    for key, val in correctness.items():
        print(str(key) + ": " + str(val / size))

test()
print("--- Search in Ball Tree with Size {} spent {} seconds ---".format(size, (time.time() - constructTime)))
print("--- Total time is {} seconds ---".format((time.time() - startTime)))