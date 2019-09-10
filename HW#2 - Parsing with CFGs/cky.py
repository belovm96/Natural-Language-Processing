"""
COMS W4705 - Natural Language Processing - Summer 19 
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg

### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        
        """
        self.grammar = grammar

    def is_in_language(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        
        """
        
        table = {}
        
        n = 0
        length = 2
        c = 0
        
        while c <= len(tokens):
            
            if c == 0:
                
                for token in tokens:
                    
                    values = self.grammar.rhs_to_rules[tuple([token])]
                    lst = []
                    for value in values:
                        lst.append(value[0])
                    table[(n,n+1)] = tuple(lst)
                            
                         
                            
                    n += 1
            
            
            else:
                
                for i in range(len(tokens)-c):
                    
                    n = 0
                    lst = []
                    
                    for j in range(1,c+1):
                        
                        value1 = table[(i,i+length-j)]
                        value2 = table[(i+length-j,i+length)]
                        
                        for key2 in value1:
                            key1 = key2
                            
                            for key3 in value2:
                                key2 = key3
                                values = self.grammar.rhs_to_rules[(key1,key2)]
                        
                                if len(values) > 0:
                            
                                    for value in values:
                                        
                                        if lst.count(value[0]) < 1:
                                            
                                            lst.append(value[0])
                                
                    if len(lst) == 0:
                    
                        table[(i,i+length)] = tuple()
                        
                    else:            
                        table[(i,i+length)] = tuple(lst)   
                  
                length += 1
                         
            c += 1
        
        
        for top in table[(0,len(tokens))]:
            
            if top == 'TOP':
                
                return True
    
        return False 
       
    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        
        """
        
        table = {}
        probs = {}
        n = 0
        length = 2
        c = 0
        
        while c <= len(tokens):
            
            if c == 0:
                
                for token in tokens:
                    
                    values = self.grammar.rhs_to_rules[tuple([token])]
                    
                    lst = {}
                    lst1 = {}
                    
                    for value in values:
                        
                        
                        lst[value[0]] = math.log2(value[2])
                        lst1[value[0]] = value[1][0]
                        
                    probs[(n,n+1)] = lst
                    table[(n,n+1)] = lst1
                            
                        
                    n += 1
                
                
            
            else:
                
                for i in range(len(tokens)-c):
                    
                    lst = {}
                    lst1 = {}
                    
                    for j in range(1,c+1):
                        
                        value1 = table[(i,i+length-j)]
                        value2 = table[(i+length-j,i+length)]
                        
                        for key3, value in value1.items():
                            key1 = key3
                        
                            for key4, value in value2.items():
                                key2 = key4
                    
                                values = self.grammar.rhs_to_rules[(key1,key2)]
                                
                                if len(values) > 0:
                            
                                    for value in values:
                        
                                        p = probs[(i,i+length-j)][key1]+probs[(i+length-j,i+length)][key2]+math.log2(value[2])
                                            
                                        if value[0] in lst:
                                                
                                            if p > lst[value[0]]:
                                                lst[value[0]] = p
                                                lst1[value[0]] = ((key1,i,i+length-j),(key2,i+length-j,i+length))
                                        else:
                                            lst[value[0]] = p
                                            lst1[value[0]] = ((key1,i,i+length-j),(key2,i+length-j,i+length))
            
                                        
                    probs[(i,i+length)] = lst
                    table[(i,i+length)] = lst1
                        
                length += 1
                            
            c += 1
        
        return table, probs


def get_tree(chart, i,j,nt): 
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    
    """
    sub_tree = chart[(i,j)][nt]
    
    if len(sub_tree) > 1 and isinstance(sub_tree, tuple):
        
        
        tree = (nt,get_tree(chart,sub_tree[0][1],sub_tree[0][2],sub_tree[0][0]),get_tree(chart,sub_tree[1][1],sub_tree[1][2],sub_tree[1][0]))
        
    else:
        
        tree = (nt,sub_tree)
        
    
    
    return tree
 
       
if __name__ == "__main__":
    
    with open('atis3.pcfg','r') as grammar_file: 
        grammar = Pcfg(grammar_file) 
        parser = CkyParser(grammar)
        toks =['flights', 'from', 'miami', 'to', 'cleveland', '.'] 
        print(parser.is_in_language(toks))
        table,probs = parser.parse_with_backpointers(toks)
        print(table)
        print(probs)
        print(get_tree(table,0,len(toks),grammar.startsymbol))
        
        print(check_probs_format(probs))
        print(check_table_format(table))
