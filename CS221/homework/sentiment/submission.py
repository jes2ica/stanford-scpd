#!/usr/bin/python

import random
import collections
import math
import sys
from util import *

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    dict = collections.defaultdict(int)
    for word in x.split():
        dict[word] += 1

    return dict

############################################################
# Problem 3b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = collections.defaultdict(float)  # feature => weight
    for _ in range(numIters):
        for train in trainExamples:
            features = featureExtractor(train[0])
            y = train[1]
            wphi = 0
            # calculate loss based on current weights
            for x in features:
                wphi += weights[x] * features[x]
            loss = max(0, (1 - wphi * y))
            # update weights only when loss > 0
            if loss > 0:
                for x in features:
                    # w -= eta * gradient
                    weights[x] += eta * features[x] * y
    return weights

############################################################
# Problem 3c: generate test case

def sparseVectorDotProduct(v1, v2):
    """
    Given two sparse vectors |v1| and |v2|, each represented as collections.defaultdict(float), return
    their dot product.
    You might find it useful to use sum() and a list comprehension.
    This function will be useful later for linear classifiers.
    """
    ans = 0
    for index, val in v1.items():
        ans += val * v2[index]
    return ans

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        phi = {}
        phi[random.choice(weights.keys())] = random.randint(1, 10)
        y = 1 if dotProduct(weights, phi) >= 0 else -1
        return (phi, y)
    return [generateExample() for _ in range(numExamples)]

############################################################
# Problem 3e: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        s = ''.join(x.split())
        features = collections.defaultdict(int)
        if len(s) < n:
            return features
        for i in range(len(s) - n + 1):
            key = s[i:i+n]
            features[key] += 1
        return features
    return extract

############################################################
# Problem 4: k-means
############################################################


def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    def distance(loc1, loc2):
        return dotProduct(loc1, loc1) + dotProduct(loc2, loc2) - 2 * dotProduct(loc1, loc2)

    def getCenter(points):
        center = {}
        keys = set([k for p in points for k in p.keys() ])
        count = len(points)
        for k in keys:
            val = sum([p[k] for p in points]) / float(count)
            center[k] = val
        return center

    assignments = []
    centers = random.sample(examples, K)
    squareLoss = 0
    prevLoss = -1

    for counter in range(maxIters):
        assignments = []
        squareLoss = 0
        # Assign points to clusters
        for point in examples:
            distances = [distance(point, c) for c in centers]
            minIndex, minDistance = min(enumerate(distances), key = lambda p: p[1])
            assignments.append(minIndex)
            squareLoss += minDistance

        if squareLoss == prevLoss:
            break;
        else:
            prevLoss = squareLoss

        # Find new centers
        newCenters = []
        for index, center in enumerate(centers):
            points = []
            for i in range(len(assignments)):
                if index == assignments[i]:
                    points.append(examples[i])
            newCenters.append(getCenter(points))
        centers = newCenters

    return centers, assignments, squareLoss
