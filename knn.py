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

size = 100

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

distantTable = [[0] * 256 for _ in range(256)]
for i in range(len(distantTable)):
    for j in range(len(distantTable[0])):
        distantTable[i][j] = (i - j) ** 2

def getDistance(image1, image2):
    sqrtSum = 0
    for i in range(28):
        for j in range(28):
            sqrtSum += (int(image1[i][j]) - image2[i][j]) ** 2
    return sqrtSum

def getCentroid(data):
    sums = [[0] * 28 for _ in range(28)]
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
        image = data[d].image
        distance = getDistance(image, target)
        if distance > furthest[0]:
            furthest = (distance, d)
    return data[furthest[1]]

def seperateTwoBalls(data, f1, f2):
    balls = [[], []]
    for d in range(len(data)):
        image = data[d].image
        distance1 = getDistance(image, f1.image)
        distance2 = getDistance(image, f2.image)
        if distance1 < distance2:
            balls[0].append(data[d])
        else:
            balls[1].append(data[d])
    return balls

def constructBallTree(training):
    if len(training) == 1:
        return BallTreeNode(training[0], 0, 0)
    else:
        n = len(training)
        centroid = ImageData(getCentroid(training), -1)
        
        f1 = getFurthest(centroid.image, training)
        f2 = getFurthest(f1.image, training)
        balls = seperateTwoBalls(training, f1, f2)

        radius = getDistance(centroid.image, f1.image)
        ballNode = BallTreeNode(centroid, radius, n)
        
        ballNode.left = constructBallTree(balls[0])
        ballNode.right = constructBallTree(balls[1])

        return ballNode

def searchBallTree(target, k, heap, ballTree, distance=-1):
    d = getDistance(target.image, ballTree.imageData.image) if distance == -1 else distance
    if heap and d - ballTree.radius >= -heap[0][0]:
        return
    elif ballTree.leafs == 0:
        lbl = ballTree.imageData.label
        heapq.heappush(heap, (-d, lbl))
        if len(heap) > k:
            heapq.heappop(heap)
    else:
        # searchBallTree(target, k, heap, ballTree.left)
        # searchBallTree(target, k, heap, ballTree.right)
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
print("--- %s seconds ---" % (time.time() - startTime))

def test():
    correctness = {key: 0 for key in ks}
    for i in range(size):
        testLbl = testLbls[i]
        knn = []
        searchBallTree(testData[i], 100, knn, root)
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
print("Size: " + str(size) + "--- %s seconds ---" % (time.time() - startTime))

