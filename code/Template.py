import numpy as np
from collections import Counter

def my_fit( words, verbose = False ):
    dt = Tree( min_leaf_size = 1, max_depth = 15 )
    dt.fit( words, verbose )
    return dt

class Tree:
    def __init__(self, min_leaf_size, max_depth, pruning_factor):
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.pruning_factor = pruning_factor
        self.root = None
        self.words = None
        
    def fit(self, words, verbose=False):
        self.words = words
        self.root = Node(depth=0, parent=None)
        self.root.fit(all_words=self.words, my_words_idx=np.arange(len(self.words)), min_leaf_size=self.min_leaf_size, max_depth=self.max_depth, pruning_factor=self.pruning_factor, verbose=verbose)
        
    def predict(self, query):
        node = self.root
        while not node.is_leaf:
            response = node.get_response(query)
            node = node.get_child(response)
        return node.get_prediction()
        
class Node:
    def __init__(self, depth, parent):
        self.depth = depth
        self.parent = parent
        self.all_words = None
        self.my_words_idx = None
        self.children = {}
        self.is_leaf = True
        self.response = None
        self.prediction = None
        self.history = []
    
    def fit(self, all_words, my_words_idx, min_leaf_size, max_depth, pruning_factor, verbose=False):
        self.all_words = all_words
        self.my_words_idx = my_words_idx
        
        if len(my_words_idx) <= min_leaf_size or self.depth >= max_depth:
            self.process_leaf(my_words_idx)
            return
        
        best_query, best_split_dict, max_info_gain = self.get_best_query(all_words, my_words_idx)
        
        if max_info_gain < pruning_factor:
            self.process_leaf(my_words_idx)
            return
        
        self.query = best_query
        self.is_leaf = False
        
        if verbose:
            print("Depth:", self.depth)
            print("Query:", self.query)
            print("Split Dict:", best_split_dict)
        
        for response, idx in best_split_dict.items():
            child = Node(depth=self.depth + 1, parent=self)
            self.children[response] = child
            child.fit(all_words, idx, min_leaf_size, max_depth, pruning_factor, verbose)
            
    def get_best_query(self, all_words, my_words_idx):
        best_query = None
        best_split_dict = None
        max_info_gain = float('-inf')
        for query in all_words:
            split_dict = {}
            for idx in my_words_idx:
                mask = self.get_mask(all_words[idx], query)
                if mask not in split_dict:
                    split_dict[mask] = []
                split_dict[mask].append(idx)
            info_gain = self.get_info_gain(len(my_words_idx), split_dict)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_query = query
                best_split_dict = split_dict
        return best_query, best_split_dict, max_info_gain
            
    def get_mask(self, word, query):
        mask = ['_' for _ in range(len(word))]
        for i in range(min(len(word), len(query))):
            if word[i] == query[i]:
                mask[i] = word[i]
        return ''.join(mask)
    
    def get_info_gain(self, len_parent, split_dict):
        child_entropy = 0
        for indices in split_dict.values():
            child_entropy -= (len(indices) / len(self.my_words_idx)) * np.log2(len(indices) / len(self.my_words_idx))
        return np.log2

def prune( self, validation_set ):
    if self.is_leaf:
        return self
    
    # Prune all the children of this node
    for response, child in self.children.items():
        self.children[ response ] = child.prune( validation_set )
    
    # Compute the loss for this node
    loss = self.get_loss( validation_set )
    
    # Compute the loss for the children
    children_loss = 0
    for child in self.children.values():
        children_loss += child.get_loss( validation_set )
    
    # If the children are better, keep the children and return them
    if children_loss < loss:
        self.is_leaf = True
        self.children = {}
        return self
    
    # Otherwise, keep this node and return it
    return self

def get_loss( self, validation_set ):
    # For leaves, we simply count the number of errors on the validation set
    if self.is_leaf:
        count = 0
        for word, label in validation_set:
            query = self.history[-1]
            predicted_label = self.predict( query, word )
            if predicted_label != label:
                count += 1
        
        return count
    
    # For non-leaves, we add the loss of the children
    loss = 0
    for child in self.children.values():
        loss += child.get_loss( validation_set )
    
    return loss

def predict( self, query, word ):
    if self.is_leaf:
        return self.my_words_idx[0]
    
    response = self.get_response( query, word )
    child = self.get_child( response )
    return child.predict( query, word )

def get_response( self, query, word ):
    # Melbot assumes that the queries are always same-length as words
    response = ""
    for i in range( len( word ) ):
        if query[i] == "_":
            response += word[i]
        else:
            response += "_"
    
    return response

def fit( self, all_words, my_words_idx, min_leaf_size, max_depth, verbose ):
    self.all_words = all_words
    self.my_words_idx = my_words_idx
    # Melbot has to make some decisions
    # If the node should be a leaf because all remaining words have same label
    # Or if the node should be a leaf because it reaches the maximum depth
    if self.should_be_leaf():
        self.is_leaf = True
        return
    
    # If the node is not a leaf, Melbot has to split it
    self.is_leaf = False
    
    # Melbot uses information gain to select the best query
    # best_query, split_dict = self.get_best_query( all_words, my_words_idx )
    best_query, best_split_dict = self.get_best_query_pruning( all_words, my_words_idx, min_leaf_size, verbose )
    self.query_idx = best_query
    self.history.append( all_words[ self.query_idx ] )
    
    # Recurse on the children
    for response, idx_list in best_split_dict.items():
        child = Node( depth = self.depth + 1, parent = self )
        self.children[ response ] = child
        child.fit( all_words, idx_list, min_leaf_size, max_depth, verbose )
        

# Implements a simple check for whether a node should be a leaf
def should_be_leaf( self ):
    # All words have the same label
    if len( set( [ self.words[ idx ][1] for idx in self.my_words_idx ] ) ) == 1:
        return True
    
    # Maximum depth reached
    if self.depth >= self.max_depth:
        return True

    # Fewer than minimum words required
    if len(self.my_words_idx) < self.min_words:
        return True

    # Reached maximum number of nodes
    if self.num_nodes >= self.max_nodes:
        return True

    # If none of the above conditions are satisfied, then the node should not be a leaf
    return False