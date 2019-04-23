import collections
import math

############################################################
# Problem 3a

def findAlphabeticallyLastWord(text):
    """
    Given a string |text|, return the word in |text| that comes last
    alphabetically (that is, the word that would appear last in a dictionary).
    A word is defined by a maximal sequence of characters without whitespaces.
    You might find max() and list comprehensions handy here.
    """
    return max(text.split(' '))

############################################################
# Problem 3b

def euclideanDistance(loc1, loc2):
    """
    Return the Euclidean distance between two locations, where the locations
    are pairs of numbers (e.g., (3, 5)).
    """
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(loc1, loc2)]))

############################################################
# Problem 3c

def mutateSentences(sentence):
    """
    Given a sentence (sequence of words), return a list of all "similar"
    sentences.
    We define a sentence to be similar to the original sentence if
      - it as the same number of words, and
      - each pair of adjacent words in the new sentence also occurs in the original sentence
        (the words within each pair should appear in the same order in the output sentence
         as they did in the orignal sentence.)
    Notes:
      - The order of the sentences you output doesn't matter.
      - You must not output duplicates.
      - Your generated sentence can use a word in the original sentence more than
        once.
    Example:
      - Input: 'the cat and the mouse'
      - Output: ['and the cat and the', 'the cat and the mouse', 'the cat and the cat', 'cat and the cat and']
                (reordered versions of this list are allowed)
    """
    # Build the graph
    graph = {}
    words = sentence.split()
    res = set()
    for i in range(len(words) - 1):
        w = words[i]
        if not w in graph:
            graph[w] = set()
        graph[w].add(words[i + 1])
    # Helper function that recursively find and append next word
    def helper(graph, length, tmp, res):
        if len(tmp) == length:
            res.add(' '.join(tmp))
            return
        cur = tmp[-1]
        if cur not in graph:
            return
        for word in graph[cur]:
            tmp.append(word)
            helper(graph, length, tmp, res)
            tmp.pop()
    # For each vertex in the graph, call the helper function
    for word in graph:
        helper(graph, len(words), [word], res)
    return res

############################################################
# Problem 3d

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

############################################################
# Problem 3e

def incrementSparseVector(v1, scale, v2):
    """
    Given two sparse vectors |v1| and |v2|, perform v1 += scale * v2.
    This function will be useful later for linear classifiers.
    """
    for index in v2:
        v1[index] += v2[index] * scale

############################################################
# Problem 3f

def findSingletonWords(text):
    """
    Splits the string |text| by whitespace and returns the set of words that
    occur exactly once.
    You might find it useful to use collections.defaultdict(int).
    """
    counter = collections.Counter(text.split(' '))
    return set(key for key in counter if counter[key] == 1)

############################################################
# Problem 3g

def computeLongestPalindromeLength(text):
    """
    A palindrome is a string that is equal to its reverse (e.g., 'ana').
    Compute the length of the longest palindrome that can be obtained by deleting
    letters from |text|.
    For example: the longest palindrome in 'animal' is 'ama'.
    Your algorithm should run in O(len(text)^2) time.
    You should first define a recurrence before you start coding.
    """
    n = len(text)
    if n == 0:
        return 0

    maxLen = [[0] * n for _ in range(n)]
    for i in reversed(range(n)):
        maxLen[i][i] = 1
        for j in range(i + 1, n):
            if text[i] == text[j]:
                maxLen[i][j] = maxLen[i+1][j-1] + 2
            else:
                maxLen[i][j] = max(maxLen[i+1][j], maxLen[i][j-1])
    return maxLen[0][n-1]
