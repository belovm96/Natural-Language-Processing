from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys
import numpy as np
import keras

from extract_training_data import FeatureExtractor, State
class Parser(object): 

    def __init__(self, extractor, modelfile):
        
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        # The following dictionary from indices to output actions will be useful
        trans = ['left_arc','right_arc']
        c = 0
        possible_pairs = {}
        dep_relations = ['tmod', 'vmod', 'csubjpass', 'rcmod', 'ccomp', 'poss', 'parataxis', 'appos', 'dep', 'iobj', 'pobj', 'mwe', 'quantmod', 'acomp', 'number', 'csubj', 'root', 'auxpass', 'prep', 'mark', 'expl', 'cc', 'npadvmod', 'prt', 'nsubj', 'advmod', 'conj', 'advcl', 'punct', 'aux', 'pcomp', 'discourse', 'nsubjpass', 'predet', 'cop', 'possessive', 'nn', 'xcomp', 'preconj', 'num', 'amod', 'dobj', 'neg','dt','det']
        for tran in trans:
            for dep_rel in dep_relations:
                
                possible_pairs[c] = (tran,dep_rel)
                
                c += 1
                
        possible_pairs[c] = ('shift',None)
        self.output_labels = possible_pairs
        
    def parse_sentence(self, words, pos):
        
        state = State(range(1,len(words)))
        
        state.stack.append(0)    
        
        while state.buffer: 
            
            arr = np.reshape(self.extractor.get_input_representation(words,pos,state),(-1,6))
            
            trans = self.model.predict(arr)
            
            trans = trans.flatten()
            
            trans_sorted = np.sort(trans)
            
            k = 1
            
            max_prob = trans_sorted[-k]
            
            ind = np.where(trans == max_prob)
            
            transition = self.output_labels[ind[0][0]] 
            
            boo = 'Not done'
            
            while boo == 'Not done':
                
                if len(state.stack) > 0 and transition[0] == 'left_arc':
                    state.left_arc(transition[1])
                    boo = 'Done'
                elif len(state.stack) > 0 and transition[0] == 'right_arc':
                    state.right_arc(transition[1])
                    boo = 'Done'
                    
                elif len(state.stack) == 0 and transition[0] == 'shift' and len(state.buffer) > 0:
                    state.shift()
                    boo = 'Done'
                    
                elif len(state.stack) > 0 and transition[0] == 'shift' and len(state.buffer) > 1:
                    state.shift()
                    boo = 'Done'
                    
                elif state.stack[-1] == 0 and transition[0] != 'left_arc':
                    state.left_arc(transition[1])
                    boo = 'Done'
                    
                else:
                    
                    k += 1
                    
                    max_prob = trans_sorted[-k]
            
                    ind = np.where(trans == max_prob)
            
                    transition = self.output_labels[ind[0][0]]
                    
        
        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, 'data/model.h5')
    
    with open('data/dev.conll','r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
     
