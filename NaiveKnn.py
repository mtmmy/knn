import load_dataset
import numpy as np
import heapq
from collections import Counter
import time

trainingData = load_dataset.read("training")
testData = load_dataset.read("testing")

size = 500

trainLbls = trainingData[0][:size * 6]
trainImgs = trainingData[1][:size * 6]
testLbls = testData[0][:size]
testImgs = testData[1][:size]
ks = [1, 3, 5, 10, 30, 50, 70, 80, 90, 100]

distanceForTests = []
trainingData = []
testData = []

for i in range(len(trainLbls)):
    image = [[0] * 28 for _ in range(28)]
    for r in range(28):
        for c in range(28):
            image[r][c] = int(trainImgs[i][r][c])
    trainingData.append(np.array(image))

for i in range(len(testLbls)):
    image = [[0] * 28 for _ in range(28)]
    for r in range(28):
        for c in range(28):
            image[r][c] = int(testImgs[i][r][c])
    testData.append(np.array(image))

startTime = time.time()

def getDistance(image1, image2):
    return np.linalg.norm(image1 - image2)

def getPrediction(knn):
    guess = [n[1] for n in knn]
    return Counter(guess).most_common(1)[0][0]

for i in range(size):
    distance = []
    for j in range(size * 6):
        tlbl = trainLbls[j]
        d = getDistance(testData[i], trainingData[j])
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
