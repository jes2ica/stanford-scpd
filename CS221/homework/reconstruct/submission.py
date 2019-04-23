import shell
import util
import wordsegUtil

############################################################
# Problem 1b: Solve the segmentation problem under a unigram model

class SegmentationProblem(util.SearchProblem):
    def __init__(self, query, unigramCost):
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        return 0

    def isEnd(self, state):
        return state == len(self.query)

    def succAndCost(self, state):
        for wordLen in range(1, len(self.query) - state + 1):
            newState = state + wordLen
            action = self.query[state: newState]
            cost = self.unigramCost(action)
            yield action, newState, cost

def segmentWords(query, unigramCost):
    if len(query) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(SegmentationProblem(query, unigramCost))
    return ' '.join(ucs.actions)

############################################################
# Problem 2b: Solve the vowel insertion problem under a bigram cost

class VowelInsertionProblem(util.SearchProblem):
    def __init__(self, queryWords, bigramCost, possibleFills):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # cur_pos, last_word
        return 0, wordsegUtil.SENTENCE_BEGIN

    def isEnd(self, state):
        return state[0] == len(self.queryWords)

    def succAndCost(self, state):
        curPos, lastWord = state
        word = self.queryWords[curPos]
        words = self.possibleFills(word)
        if len(words) == 0:
            words = [self.queryWords[curPos]]
        for word in words:
            newState = curPos + 1, word
            action = word
            cost = self.bigramCost(lastWord, word)
            yield action, newState, cost

def insertVowels(queryWords, bigramCost, possibleFills):
    if len(queryWords) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(VowelInsertionProblem(queryWords, bigramCost, possibleFills))
    return ' '.join(ucs.actions)

############################################################
# Problem 3b: Solve the joint segmentation-and-insertion problem

class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(self, query, bigramCost, possibleFills):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # cur_pos, last_word
        return 0, wordsegUtil.SENTENCE_BEGIN

    def isEnd(self, state):
        return state[0] == len(self.query)

    def succAndCost(self, state):
        curPos, lastWord = state
        for wordLen in range(1, len(self.query) - curPos + 1):
            words = self.possibleFills(self.query[curPos: curPos + wordLen])
            for word in words:
                newState = curPos + wordLen, word
                action = word
                cost = self.bigramCost(lastWord, word)
                yield action, newState, cost

def segmentAndInsert(query, bigramCost, possibleFills):
    if len(query) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(JointSegmentationInsertionProblem(query, bigramCost, possibleFills))
    return ' '.join(ucs.actions)

############################################################

if __name__ == '__main__':
    shell.main()
