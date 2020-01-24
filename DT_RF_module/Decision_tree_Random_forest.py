#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 14:11:36 2020

@author: anindya
"""
import math,random,statistics

class DecisionTree():
    def __init__(self):
        self.depth = 0
        
    def _find_max_depth(self,data):
        """
        find the maximum depth of a tree
        """
        return int((len(data)-1)*len(data[0]))

    def _overall_purity(self,predicted_labels, original_labels):
        """
        compute overall purity from predicted labels and original labels
        input: 
            predicted_labels: list of labels after computed cutoff
            original_labels: list of labels for the fitting data
        return: 
            overall purity computed from left and right nodes
        sub_fun: 
            _each_node_purity(node)
            _purity(node)
        """
        def _each_node_purity(node):
            """
            input: 
                node: list of labels from a single tree node for purity computation
            returns: 
                purity value for a given node
            sub_fun: 
                _purity(node)
            """
            def _purity(labels):
                if type(labels)!=list: labels=list(labels)
                zeros, ones = labels.count(0), labels.count(1)
                return max(zeros, ones) / len(labels)

            purity = 0.0
            if len(node)>0: 
                purity = _purity(node) # compute purity for further nodes        
            return purity # purity value for an individal node
        
        # purity for left node (i.e, all True)
        left_indx=[x 
                   for ii,xx in enumerate(predicted_labels) 
                   for i,x in enumerate(original_labels) 
                   if xx if i==ii]
        left_purity= _each_node_purity(left_indx) 
        
        #purity for right node (i.e, all false)
        right_indx=[x 
                    for ii,xx in enumerate(predicted_labels)
                    for i,x in enumerate(original_labels)
                    if not xx if i==ii]
        
        right_purity = _each_node_purity(right_indx)
        return 1.0 - (left_purity+right_purity) #final purity 
    
    def _best_split_of_features(self, data,original_labels):
        """
        Find the best split from all features
        input: 
            data: list of data samples and features 
            original_labels: list of corresponding labels of given data
        returns: 
            the column to split on, 
            the cutoff value, and 
            the overall purity
        sub_fun: 
            _overall_purity( predicted_labels, original_labels)
            _each_node_purity(node)
            _purity(node)
        """       
        idx = None
        overall_max_purity = math.inf# define inital max_purity value to compare with
        cutoff = None
        data=list(map(list, zip(*data)))# transpose datapoints on the column to avoid another for loop
        for feature_index, row in enumerate(data):  # iterating through each feature
            max_purity=math.inf
            row_cutoff=None
            for value in set(row):
                # separate data samples into 2 groups(true and false)  
                predict_labels = [1 if i<value else 0 for i in row]
                compute_purity = self._overall_purity(predict_labels , original_labels)
                if compute_purity <= max_purity:  # check if it's the best one so far
                    max_purity = compute_purity # update it as best purity value for a column 
                    row_cutoff = value #store the best cutoff for a given column
            if max_purity == 0:    # find the first perfect cutoff. Stop Iterating
                return feature_index, row_cutoff, max_purity
            elif max_purity <= overall_max_purity:  # check if it's best so far
                overall_max_purity = max_purity # update it as best purity value overall
                idx = feature_index # index of the iterating column of feature 
                cutoff = row_cutoff # assign it as a best cutoff value obtained overall
        return idx, cutoff, overall_max_purity

    def fit(self, data, original_labels, tree={}, depth=0,max_depth=None):
        """
        function to fit tree for given training data
        input:
            sample: list of Feature set
            labels: list of target variable
            tree: dictinoary of tree generated for given data and original_labels. 
            depth: the depth of the current layer
            max_depth: Maximum depth of the tree
        return: 
            tree in the form of a dictionary
        sub_fun: 
            _best_split_of_features(data,original_labels)
        """
        # Find overall max_depth
        if depth==0 and max_depth==None: max_depth=self._find_max_depth(data)
        # base case 1: tree stops at previous level
        if tree is None:  return None 
         # base case 2: no data in this group
        elif len(original_labels) == 0: return None 
        # base case 3: all y is the same in this group
        elif len(set(original_labels))<2:   
            self.trees = {'label':original_labels[0]}
            return self.trees
        # base case 4: max depth reached 
        elif depth >= max_depth: return None   
        # Recursively generate trees!
        else:                
            feature_index, cutoff, purity = self._best_split_of_features(data,original_labels)  
            
            #build the first node of a tree and save it as a dictionary
            tree = {'index_col':feature_index,
                    'cutoff':cutoff,
                    'purity':purity,
                    'label': int(round(statistics.mean(original_labels)))} 
            
            node_data=[n[feature_index] for i,n in enumerate(data)]
            #get left tree data and labels
            left_node_data,left_node_label=map(list,
                                    (zip(*[(a,b) 
                                    for i, (a,b) in enumerate(zip(data,original_labels)) 
                                           if node_data[i]<cutoff ])))

            tree['left'] = self.fit(left_node_data,
                                    left_node_label,
                                        tree={}, 
                                        depth=depth+1,
                                        max_depth=max_depth)
            #get right tree data and labels
            right_node_data,right_node_label=map(list,
                                    (zip(*[(a,b) 
                                    for i, (a,b) in enumerate(zip(data,original_labels)) 
                                    if not node_data[i]<cutoff ])))
            
            tree['right'] = self.fit(right_node_data,
                                     right_node_label,
                                     tree={}, 
                                     depth=depth+1,
                                     max_depth=max_depth)  
            self.depth += 1   # increase the depth since purity function call fit once
            self.trees = tree  
            return tree
    
    def predict(self, data):
        """
        Function to predict using a fitted tree
        input:
            data: list of data-points or a single data-point
        return:
            predictions for all given data-points of a single data-point
        """
        if not any(isinstance(i, list) for i in data): data=[data] #check if its a single sample
        predictions = []*len(data)# list to store results
        for feature_index, row in enumerate(data):
            cur_layer = self.trees
            while cur_layer.get('cutoff'): #iterate until no more cutoff vlaue
                if row[cur_layer['index_col']] < cur_layer['cutoff']:
                    cur_layer = cur_layer['left']
                else:
                    cur_layer = cur_layer['right']
                    
            else: predictions.append(cur_layer.get('label'))
       
        # to get a prediction for single row
        if len(predictions)<2: 
            predictions= predictions[0]
        return predictions

class RandomForest:
    def __init__(self, n_trees=None,ratio=None):
        self.n_trees=n_trees #minimum number of trees
        self.ratio=ratio #ratio for sub-sampling of data
        self.trees = list() #store dictinoaries of trees in a list
        if self.ratio==None: self.ratio=70 # suggested in the assignment
            
    def fit(self, data,labels):
        """
        fit n trees for randomly sampled data
        input:
            data: list of samples
            labels: list of labels for the given data
        return:
            list consisting of n_trees as dictionary
        sub_fun:
            _subsample(data,labels)
        """
        def _subsample(data,labels):
            """
            generator for simple random sampling without replacement
            input:
                data: list of datapoints 
                labels: list of corresponding labels
            return:
               a generated list of sub-sampled data and their correspoind labels 
            """
            #empty lists to load sub_sampled data and labels
            sub_sample_data,sub_sample_labels = list(),list()
            if type(data)!=list: data=list(data)#check if type of data is list or not
            
            # get the value of total number for subsampling
            n_sample = round(len(data) * self.ratio/100)
            while len(sub_sample_data) < n_sample:
                index = random.randrange(len(data))# get the random indexes to pick data-points and labels
                sub_sample_data.append(data[index]) #store the sub-sampled data-points
                sub_sample_labels.append(labels[index])#store the labels for sub-sampled data-points
            yield sub_sample_data,sub_sample_labels #genrator for yielding sampled data-points 

        for i in range(self.n_trees):  
            # extract data-points and labels from a tuple of lists
            sub_sample_data= list(_subsample(data,labels))[0]
            tree = DecisionTree()
            tree.fit(sub_sample_data[0],sub_sample_data[1])# start fitting trees
            self.trees.append(tree) #store the dictinoaries of trees in a list
        return self.trees

    def predict(self, data):
        """
        Function to predict using bagging of fitted trees based on majority voting
        input:
            data: list of data-points or a single data-point
        return:
            predictions for all given data-points of a single data-point
        sub_fun:
            _baggingPrediction(row)
        """
        
        def _baggingPrediction(row):
            """
            make a prediction with a list of bagged trees using majority voting
            input:
                row: a single datapoint from the list of data-points
            return:
                prediction from n trees based on majority voting
            """
            #compute labels for each data-point using 'n' fitted trees
            predictions = [tree.predict(row) for tree in self.trees] 
            #predict the labels using majority voting 
            return max(set(predictions), key=predictions.count)
        
        if not any(isinstance(i, list) for i in data): data=[data]#check if its a single sample
        predictions = [_baggingPrediction(row) for row in data] #predicted label
        
        # to get a prediction for single row
        if len(predictions)<2: 
            predictions=predictions[0]
        return predictions

if __name__ == '__main__':
    DecisionTree()
    RandomForest()