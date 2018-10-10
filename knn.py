import load_dataset
import numpy as np
import heapq
from collections import Counter
import time

trainingData = load_dataset.read("training")
testData = load_dataset.read("testing")

size = 100

trainLbls = trainingData[0][:size * 6]
trainImgs = trainingData[1][:size * 6]
testLbls = testData[0][:size]
testImgs = testData[1][:size]
ks = [1, 3, 5, 10, 30, 50, 70, 80, 90, 100]

distanceForTests = []
distantTable = [[0] * 256 for _ in range(256)]

startTime = time.time()

for i in range(len(distantTable)):
    for j in range(len(distantTable[0])):
        distantTable[i][j] = (i - j) ** 2


shrink = 5
def euclideanDistance(image1, image2):    
    image1 = image1[shrink:-shrink, shrink:-shrink]
    image2 = image2[shrink:-shrink, shrink:-shrink]
    m, n = len(image1), len(image1[0])
    sqrtSum = 0

    for i in range(m):
        for j in range(n):
            sqrtSum += distantTable[image1[i][j]][image2[i][j]]
    
    return sqrtSum      # no need to square root if only doing comparison

def getPrediction(knn):
    guess = [n[1] for n in knn]
    return Counter(guess).most_common(1)[0][0]

for i in range(size):
    distance = []
    for j in range(size * 6):
        tlbl = trainLbls[j]
        d = euclideanDistance(testImgs[i], trainImgs[j])
        heapq.heappush(distance, (d, tlbl))
    distanceForTests.append(distance)

print("--- %s seconds ---" % (time.time() - startTime))

for k in ks:
    corrects = 0
    for i in range(size):
        testLbl = testLbls[i]

        heap = distanceForTests[i]
        knn = heapq.nsmallest(k, heap)
        if getPrediction(knn) == testLbl:
            corrects += 1
    print(corrects / size)
    
print("--- %s seconds ---" % (time.time() - startTime))

