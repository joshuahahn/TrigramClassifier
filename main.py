from collections import defaultdict
import math

# Returns a generator of the sentences in the script.
def file_reader(file, lexicon=None):
    with open(file, 'r') as script:
        for line in script:
            sequence = line.lower().strip().split()
            if line == '\n' or line == ' \n':
                continue
            if lexicon:
                res = []
                for word in sequence:
                    if word in lexicon:
                        res.append(word)
                    else:
                        res.append('UNK')
                yield res
            else:
                yield sequence

# Given a corpus, this will generate the lexicon of the text.
def generate_lexicon(corpus):
    word_frequency = defaultdict(int)
    for line in corpus:
        for word in line:
            word_frequency[word] += 1
    return set(word for word in word_frequency if word_frequency[word] > 1)

def get_ngrams(sequence, n):
    # Data-pre-processing; pad the sequence with "START"s and a "STOP".
    for idx in range(n - 1):
        sequence.insert(0,'START')
    sequence.append('STOP')

    if n == 1: # Edge case; if n == 1, then 'START' will not be prepended to sequence.
        sequence.insert(0,'START')

    res = []
    # The number of items in res will be len(sequence) - n + 1.
    for idx in range(len(sequence) - n + 1):
        ngram = []
        for tupleItem in range(n):
            ngram.append(sequence[idx + tupleItem])
        res.append(tuple(ngram))
    return res

# We create a character model for each main cast member.
class CharacterModel(object):
    
    def __init__(self, corpus, character):
        generator = file_reader(corpus)
        self.lexicon = generate_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
        self.type = 0
        self.token = 0
        self.numSentences = 0

        self.character = character + ':'

        generator = file_reader(corpus, self.lexicon)
        self.count_ngrams(generator)

    def count_ngrams(self, corpus):
        self.unigramcounts = {} 
        self.bigramcounts = {} 
        self.trigramcounts = {} 

        for sentence in corpus:
            if sentence == '\n':
                continue
            # Ignore sentence if this is not the character we are interested in.
            if sentence[0] != self.character:
                continue
            
            # Ignore sentence if it begins with (
            if sentence[0][0] == '(' or sentence[0] [0] == '[':
                continue

            # Remove the cast name indicating who is speaking.
            sentence = sentence[1:]

            self.numSentences = self.numSentences + 1
            unigrams = get_ngrams(sentence, 1)
            bigrams = get_ngrams(sentence, 2)
            trigrams = get_ngrams(sentence, 3)

            # Update unigrams
            for unigram in unigrams:
                if unigram in self.unigramcounts.keys():
                    self.unigramcounts[unigram] = self.unigramcounts[unigram] + 1
                else:
                    self.unigramcounts[unigram] = 1
                    self.type = self.type + 1 # update size of lexicon
                self.token = self.token + 1 # update number of unigrams seen
            
            # Update bigrams
            for bigram in bigrams:
                if bigram in self.bigramcounts.keys():
                    self.bigramcounts[bigram] = self.bigramcounts[bigram] + 1
                else:
                    self.bigramcounts[bigram] = 1
            
            # Update trigrams
            for trigram in trigrams:
                if trigram in self.trigramcounts.keys():
                    self.trigramcounts[trigram] = self.trigramcounts[trigram] + 1
                else:
                    self.trigramcounts[trigram] = 1
        return 

    def raw_trigram_probability(self,trigram):
        # Trigram = (a,b,c)
        # Probability of trigram = P(c|a,b)
        # = count(trigram) / count(a,b)

        # Edge case: P(word | start, start)
        if trigram[0] == 'START' and trigram[1] == 'START':
            if trigram in self.trigramcounts.keys():
                return self.trigramcounts[trigram] / self.numSentences
            return 0.0

        if not trigram[:2] in self.bigramcounts.keys():
            return self.raw_unigram_probability((trigram[2],))

        if not trigram in self.trigramcounts.keys():
            return 0.0

        return self.trigramcounts[trigram] / self.bigramcounts[trigram[:2]]

    def raw_bigram_probability(self, bigram):
        # P(bigram(a,b)) = P(b|a)
        # P(b|a) = count(a,b) / count(a)
        
        # bigram is a tuple.
        if not bigram in self.bigramcounts.keys():
            return 0
        return self.bigramcounts[bigram] / self.unigramcounts[(bigram[0],)]
    
    def raw_unigram_probability(self, unigram):
        if not unigram in self.unigramcounts.keys():
            return 1 / self.token
        return self.unigramcounts[unigram] / self.token

    def smoothed_trigram_probability(self, trigram):
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        res = lambda1 * self.raw_trigram_probability(trigram)
        res = res + lambda2 * self.raw_bigram_probability(tuple(trigram[1:]))
        res = res + lambda3 * self.raw_unigram_probability((trigram[2],))
        return res
        
    def sentence_logprob(self, sentence):
        trigrams = get_ngrams(sentence,3)
        res = 0.0
        for trigram in trigrams:
            res = res + math.log2(self.smoothed_trigram_probability(trigram))
        return res

    def perplexity(self, corpus):
        res = 0.0
        M = 0
        for sentence in corpus:
            res = res + self.sentence_logprob(sentence)
            for word in sentence:
                if (word != 'START'):
                    M = M + 1
        res = res / M
        res = res * -1
        res = 2**res

        return res


if __name__ == '__main__':
    JoeyModel = CharacterModel('Friends_Transcript.txt', 'joey')
    print('Joey, number of lines: ' + str(JoeyModel.numSentences))
    ChandlerModel = CharacterModel('Friends_Transcript.txt', 'chandler')
    print('Chandler, number of lines: ' + str(ChandlerModel.numSentences))
    RossModel = CharacterModel('Friends_Transcript.txt', 'ross')
    print('Ross, number of lines: ' + str(RossModel.numSentences))
    RachelModel = CharacterModel('Friends_Transcript.txt', 'rachel')
    print('Rachel, number of lines: ' + str(RachelModel.numSentences))
    PhoebeModel = CharacterModel('Friends_Transcript.txt', 'phoebe')
    print('Phoebe, number of lines: ' + str(PhoebeModel.numSentences))
    MonicaModel = CharacterModel('Friends_Transcript.txt', 'monica')
    print('Monica, number of lines: ' + str(MonicaModel.numSentences))

    # Let's make some predictions.
    # Let's see who is most likely to say: "Ooh, here's that macadamida nut!"
    # Answer: Ross
    test1ross = [['ooh,', "here's", 'that', 'macadamia', 'nut!']]

    print(" ")
    print("Test 1: 'Ooh, here's that macadamia nut!'")
    print("Said by: Ross")

    ppJoey = JoeyModel.perplexity(test1ross)
    print("Perplexity, Joey: " + str(ppJoey))
    ppChandler = ChandlerModel.perplexity(test1ross)
    print("Perplexity, Chandler: " + str(ppChandler))
    ppRoss = RossModel.perplexity(test1ross)
    print("Perplexity, Ross: " + str(ppRoss))
    ppRachel = RachelModel.perplexity(test1ross)
    print("Perplexity, Rachel: " + str(ppRachel))
    ppPhoebe = PhoebeModel.perplexity(test1ross)
    print("Perplexity, Phoebe: " + str(ppPhoebe))
    ppMonica = MonicaModel.perplexity(test1ross)
    print("Perplexity, Monica: " + str(ppMonica))

    pps = [ppJoey, ppChandler, ppRoss, ppRachel, ppPhoebe, ppMonica]
    names = ['Joey', 'Chandler', 'Ross', 'Rachel', 'Phoebe', 'Monica']
    predicted = names[min(range(len(pps)), key=pps.__getitem__)]
    print("Predicted: " + predicted)

    
    # Next, let's see who's most likely to say: "There's nothing to tell! He's just some guy I work with!"
    # This also happens to be the first line in the Friends TV show, and the line is by Monica.

    print(" ")
    print("Test 2: 'There's nothing to tell! He's just some guy I work with!'")
    print("Said by: Monica")

    test2monica = [["there's", 'nothing", "to', "tell!", "He's", "just", "some", "guy", "I", "work", "with!"]]
    ppJoey = JoeyModel.perplexity(test2monica)
    print("Perplexity, Joey: " + str(ppJoey))
    ppChandler = ChandlerModel.perplexity(test2monica)
    print("Perplexity, Chandler: " + str(ppChandler))
    ppRoss = RossModel.perplexity(test2monica)
    print("Perplexity, Ross: " + str(ppRoss))
    ppRachel = RachelModel.perplexity(test2monica)
    print("Perplexity, Rachel: " + str(ppRachel))
    ppPhoebe = PhoebeModel.perplexity(test2monica)
    print("Perplexity, Phoebe: " + str(ppPhoebe))
    ppMonica = MonicaModel.perplexity(test2monica)
    print("Perplexity, Monica: " + str(ppMonica))

    pps = [ppJoey, ppChandler, ppRoss, ppRachel, ppPhoebe, ppMonica]
    predicted = names[min(range(len(pps)), key=pps.__getitem__)]
    print("Predicted: " + predicted)

    # Finally, let's see who is most likely to say "I love you."
    # Ross has said this before, but there have also been many other instances.

    print(" ")
    print("Test 3: 'I love you.'")
    print("Said by: All")

    test3ross = [["i", "love", "you"]]
    ppJoey = JoeyModel.perplexity(test3ross)
    print("Perplexity, Joey: " + str(ppJoey))
    ppChandler = ChandlerModel.perplexity(test3ross)
    print("Perplexity, Chandler: " + str(ppChandler))
    ppRoss = RossModel.perplexity(test3ross)
    print("Perplexity, Ross: " + str(ppRoss))
    ppRachel = RachelModel.perplexity(test3ross)
    print("Perplexity, Rachel: " + str(ppRachel))
    ppPhoebe = PhoebeModel.perplexity(test3ross)
    print("Perplexity, Phoebe: " + str(ppPhoebe))
    ppMonica = MonicaModel.perplexity(test3ross)
    print("Perplexity, Monica: " + str(ppMonica))

    pps = [ppJoey, ppChandler, ppRoss, ppRachel, ppPhoebe, ppMonica]
    predicted = names[min(range(len(pps)), key=pps.__getitem__)]
    print("Predicted: " + predicted)