import sys
from collections import defaultdict
import math
import random
import os
import os.path
import numpy as np

"""
COMS W4705 - Natural Language Processing - Summer 2019 
Homework 1 - Trigram Language Models
Daniel Bauer

"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    
    """
   
    start = 'START'
    k = n
    c = 0
    if n == 1:
        sequence.insert(0,start)
    while n > 1:
        
        sequence.insert(0,start)
        n = n - 1
    
    sequence.append('STOP')
    
    n_gram_list = []
    
    if k >= 1 and k < len(sequence):
        while(k <= len(sequence)):
            if k == len(sequence):
                k = None
            n_gram_list.append(tuple(sequence[c:k]))
            if k == None:
                k = len(sequence)
            c += 1
            k += 1
    
    return n_gram_list


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
        
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)
        

    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        
        """
   
        self.unigramcounts = {} # might want to use defaultdict or Counter instead
        self.bigramcounts = {} 
        self.trigramcounts = {} 
        self.all_unigrams = 0
        ##Your code here
        ngrams_list = []
        count_ngrams = 0
        
        for sequence in corpus:
            
            ngrams_list = get_ngrams(sequence,1)
            
            for ngram in ngrams_list:
                if ngram in self.unigramcounts:
                    count_ngrams = self.unigramcounts.get(ngram)
                    self.unigramcounts[ngram] = count_ngrams + 1
                    self.all_unigrams += 1
                else:
                    self.unigramcounts[ngram] = 1
                    self.all_unigrams += 1
                    
            sequence = sequence[1:-1]   
            
            ngrams_list = get_ngrams(sequence,2)
            

            for ngram in ngrams_list:
                if ngram in self.bigramcounts:
                    count_ngrams = self.bigramcounts.get(ngram)
                    self.bigramcounts[ngram] = count_ngrams + 1
                else:
                    self.bigramcounts[ngram] = 1
                    
            sequence = sequence[1:-1]
            
            ngrams_list = get_ngrams(sequence,3)
            

            for ngram in ngrams_list:
                if ngram in self.trigramcounts:
                    count_ngrams = self.trigramcounts.get(ngram)
                    self.trigramcounts[ngram] = count_ngrams + 1
                else:
                    self.trigramcounts[ngram] = 1
                    
        
        
        return self.unigramcounts, self.bigramcounts, self.trigramcounts, self.all_unigrams

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        
        """
        trigram_prob = 0.0
        trigram_count = 0.0
        prev_word_count = 0.0
        start_tuple = ('START','START')
        
        if trigram in self.trigramcounts:
            trigram_count = self.trigramcounts[trigram]
            
        prev_word = trigram[0:2]
       
        if prev_word in self.bigramcounts:
            
            prev_word_count = self.bigramcounts[prev_word]
           
            trigram_prob = trigram_count/prev_word_count
            
        else:
            
            if prev_word == start_tuple:
                
                prev_word_count = self.unigramcounts[tuple([trigram[0]])] 
                trigram_prob = trigram_count/prev_word_count
               
            else:
                trigram_prob = 1/float(len(self.unigramcounts) - 1) 
                
            
        return trigram_prob

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        
        """
        
        bigram_prob = 0.0
        bigram_count = 0.0
        prev_word_count = 0.0
        
        if bigram in self.bigramcounts:
            bigram_count = self.bigramcounts[bigram]
            
        prev_word = tuple([bigram[0]])
        
        if prev_word in self.unigramcounts:
            
            prev_word_count = self.unigramcounts[prev_word]
            bigram_prob = bigram_count/prev_word_count
            
        else:
            bigram_prob = 1/float(len(self.unigramcounts) - 1)
            
            
        return bigram_prob
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        
        """
        unigram_prob = 0.0
        unigram_count = 0.0
        
        if tuple([unigram]) in self.unigramcounts:
            
            unigram_count = self.unigramcounts[tuple([unigram])]
            
            unigram_prob = unigram_count/float(self.all_unigrams)
            
        else:
            unigram_prob = 1/float(self.all_unigrams)
        
        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        return unigram_prob

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        
        """
        result = ['START','START']
        
        p = 0.0
        c = 0
        word_prob = []
        prob = []
        
        while c < t and result[-1] != 'STOP':
            
            word_prob = []
            prob = []
            if c == 0:
                
                bi = tuple(result)
            
                
            else:
                
                bi = tuple(result[-2:])
            
            for tri in self.trigramcounts:
                    
                if tri[0:2] == bi:
                
                    p = self.raw_trigram_probability(tri)
                
                    prob.append(p)
                
                    word_prob.append(tri[-1])
                    
            c += 1
            
            prediction = np.random.choice(word_prob, p=prob)
            result.append(prediction)
                   
                
        result = result[2:None]            
            
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        smoothed_tri_prob = 0.0
        
        smoothed_tri_prob = lambda1*self.raw_trigram_probability(trigram) + lambda2*self.raw_bigram_probability(trigram[1:None]) + lambda3*self.raw_unigram_probability(trigram[-1])
        
        return smoothed_tri_prob
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        
        """
        ngrams = get_ngrams(sentence,3)
       
        log_prob = 0.0
        for ngram in ngrams:
            if self.smoothed_trigram_probability(ngram) == 0.0:
                log_prob += 0.0
                
            else:
                log_prob += math.log2(self.smoothed_trigram_probability(ngram))
        
        
        return log_prob

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        
        """
        
        
        perplexity_val = 0.0
        sentence_prob = 0.0
        unigrams_all = 0
        
        for sentence in corpus:
            
            sentence_prob = self.sentence_logprob(sentence)
           
            perplexity_val += sentence_prob
            
            for word in sentence:
                
                unigrams_all += 1
            
        perplexity_val = perplexity_val/unigrams_all
        
        perplexity_val = perplexity_val*-1
        
        perplexity_val = 2**perplexity_val
        
                
        
        return perplexity_val


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

	#COMPLETE THIS FUNCTION...

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0.0
        correct = 0.0    
        acc = 0.0
        
        for f in os.listdir(testdir1):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            
            if pp1 < pp2:
                correct += 1
            total += 1
                
        for f in os.listdir(testdir2):
            pp1 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp2 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            
            if pp1 < pp2:
                correct += 1
            total += 1
            
        acc = (correct/total) * 100
        
        return acc

if __name__ == "__main__":

    model = TrigramModel('brown_train.txt') 
    
    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 
    
    print(model.generate_sentence())
    
    # Testing perplexity: 
    dev_corpus = corpus_reader('brown_test.txt', model.lexicon)
    pp = model.perplexity(dev_corpus)
    print(pp)


    # Essay scoring experiment: 
    acc = essay_scoring_experiment('train_high.txt', 'train_low.txt', 'test_high', 'test_low')
    print(acc)
    
    