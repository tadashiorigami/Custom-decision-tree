# Copyright 2022 – present by Tadashi Mori
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pandas as pd
import numpy as np
from numpy import searchsorted

# Function to calculate the profits of a transaction
def profit(transaction_values, true_y, predicted_y, profit_rate, loss_rate):
    d = {'values':transaction_values, 'true_y':true_y, 'predicted_y':predicted_y}
    dataframe = pd.DataFrame(data = d)
    profits = dataframe[(dataframe.true_y == 0) & (dataframe.predicted_y == 0)].values.sum()*profit_rate
    losses = dataframe[(dataframe.true_y == 1) & (dataframe.predicted_y == 0)].values.sum()*loss_rate
    return (profits-losses)

def get_profit_percentage(transaction_values, true_y, predicted_y, profit_rate, loss_rate):
    max_profit = profit(transaction_values, true_y, true_y, profit_rate, loss_rate)
    min_profit = profit(transaction_values, true_y, 1-true_y, profit_rate, loss_rate)
    real_profit = profit(transaction_values, true_y, predicted_y, profit_rate, loss_rate)
    return real_profit/(max_profit-min_profit)


#### Custom Decision tree
# Node information class
class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None,info_gain=None, value=None):
        '''constructor'''
        
        #for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        #for leaf node
        self.value = value


class ProfitDecisionTreeClassifier():
    def __init__(self, profit_rate , loss_rate, min_samples_split=10, max_depth=10):
        # profit_rate = a value in [0,1] to represent how much you profit in a true negative decision
        # loss_rate = a value in [0,1] to represent how much you lose in a false negative decision
        # min_samples_split = minimum samples needed for a split
        # max_depth = Maximum depth of the tree
        
        # initialize the root of the tree 
        self.root = None
        self.profit_rate = profit_rate
        self.loss_rate = loss_rate
        
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def build_tree(self, X, y, indices, amounts, curr_depth=0):
        ''' recursive function to build the tree ''' 
            
        X_ = X.loc[indices]
        y_ = y.loc[indices]
        amounts_ = amounts.loc[indices]
        num_samples = X_.shape[0]
        
        # split until stopping conditions are met
            
        if (num_samples>=self.min_samples_split) & (curr_depth<=self.max_depth):
            
            # find the best split            
            best_split = self.get_best_split(X_ , y_ , num_samples, amounts_, self.profit_rate, self.loss_rate)
            
            # check if information gain is positive
            if best_split["info_gain"]>0:
                left_indices = best_split["dataset_left"]
                right_indices = best_split["dataset_right"]
                # recur left
                left_subtree = self.build_tree(X_, y_, left_indices, amounts_, curr_depth+1)
                # recur right
                right_subtree = self.build_tree(X_, y_, right_indices, amounts_, curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        
        # compute leaf node
        leaf_value = self.calculate_leaf_value(amounts_, y_, self.profit_rate, self.loss_rate)
        # return leaf node
        return Node(value=leaf_value)
    
    def get_best_split(self, X_ , y_ , num_samples, amounts_, profit_rate, loss_rate):
        ''' function to find the best split '''
        
        # dictionary to store the best split
        features = X_.columns
        best_split = {}
        best_percentage = -float("inf")
        
        # loop over all the features
        for feature in features:
            # loop over all the feature values present in the data
            
            new_div = self.compute_profit_scores(amounts_, X_[feature], y_, profit_rate, loss_rate)
            div_threshold = self.check_best_profit_div(new_div)
            if div_threshold[3] > best_percentage:
                best_percentage = div_threshold[3]
                best_variable = feature
                best_div = div_threshold[0]
            
        # cut the data at the best threshold
        dataset_left_index = X_[X_[best_variable] < best_div].index
        dataset_right_index = X_[X_[best_variable] >= best_div].index
            
        # check if childs are not null
        if len(dataset_left_index)>0 and len(dataset_right_index)>0:
            # compute information gain
            best_split["feature_index"] = best_variable
            best_split["threshold"] = best_div
            best_split["dataset_left"] = dataset_left_index
            best_split["dataset_right"] = dataset_right_index
            best_split["info_gain"] = best_percentage
        else:
            best_split["info_gain"] = 0

        # return best split
        return best_split
    
    
    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''
        dataset_left_index = np.where(dataset[:,feature_index]<threshold)
        dataset_right_index = np.where(dataset[:,feature_index]>=threshold)
        return dataset_left_index, dataset_right_index
    
    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        ''' function to compute information gain '''
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode=="gini":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
        
    def calculate_leaf_value(self,amounts_, y_, profit_rate, loss_rate):
        ''' function to compute leaf node '''
        
        profitability = self.check_profitability(amounts_, y_, profit_rate, loss_rate)
        profits = profitability[0] + profitability[1]
        if profits <= 0:
            return 1
        if profits > 0:
            return 0    
    
    def print_tree(self, tree=None, depth=0, lines = None):
        ''' function to print the tree '''
        indent = ""
        if not tree:
            tree = self.root
            
        if not lines:
            lines = []
        line_l = lines.copy()
        line_r = lines.copy()
        line_l.append(1)
        line_r.append(0)
        
        for i in range(depth):
            if lines[i] == 0:
                indent = indent + "    "
            else:
                indent = indent + "│   "

        if tree.value is not None:
            print(tree.value)

        else:
            print(f"{tree.feature_index} < {tree.threshold}  --- Profit gain = {tree.info_gain}")
            print(f"{indent}├───left:", end="")
            self.print_tree(tree.left, depth+1,line_l)
            print(f"{indent}└───right:", end="")
            self.print_tree(tree.right, depth+1,line_r)
    
    def fit(self, X, Y, amounts):
        ''' function to train the tree '''

        indices = X.index
        self.root = self.build_tree(X,Y,indices,amounts)

    
    def make_logic_list(self, X, tree, full_logic_list, leaf_logic_list):
        if tree.value is not None:
            leaf_logic_list.append(tree.value)
        else:
            leaf_logic_list_right = leaf_logic_list.copy()
            
            leaf_logic_list.append((X[tree.feature_index] < tree.threshold))
            leaf_logic_list_right.append((X[tree.feature_index] >= tree.threshold))
            
            full_logic_list.append(leaf_logic_list_right)
            
            self.make_logic_list( X, tree.left, full_logic_list, leaf_logic_list)
            self.make_logic_list( X, tree.right, full_logic_list, leaf_logic_list_right)
    
    def predict(self, X):
        ''' function to predict new dataset '''
        
        tree = self.root
        full_logic_list = []
        leaf_logic_list= []
        full_logic_list.append(leaf_logic_list)
        self.make_logic_list(X, tree, full_logic_list, leaf_logic_list)
        prediction = pd.Series(0, index=X.index)
        
        for leaf_logics in full_logic_list:
            new_logic = leaf_logics[0]
            for logic in leaf_logics[1:-1]:
                new_logic = new_logic & logic
            prediction[new_logic] = leaf_logics[-1]
        return prediction
    
    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''
        if tree.value!=None: return tree.value
        
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
    
    def check_profitability(self, x_transactions, y_data, profit_rate, loss_rate):
        # Gets the transactions values and labels and returns percentage of profitability and profitability amount
    
        profits = x_transactions.loc[y_data[y_data == 0].index].sum() * profit_rate
        losses = x_transactions.loc[y_data[y_data == 1].index].sum() * loss_rate * -1
    
        return [profits, losses]
    
    def compute_profit_scores(self, x_transactions, x_parameter, y_sample, profit_rate, loss_rate):  
    
        scores_list = []

        final_value = x_transactions.copy()
        final_value[y_sample==0] = final_value[y_sample==0]*profit_rate
        final_value[y_sample==1] = final_value[y_sample==1]*loss_rate*-1
        max_profit = final_value[y_sample==0].sum()
        root_profit = final_value.sum()
        
        parameter_min = x_parameter.min()
        parameter_max = x_parameter.max()

        full_list = np.stack((x_parameter, final_value, y_sample), axis=1)
        full_list = full_list[full_list[:, 0].argsort()]

        if max_profit == 0:
            return [[0,0,0]]
        if root_profit<=0:
            root_profit = 0

        full_range = parameter_max - parameter_min
        first_div = 40
        second_div = 200
        first_step = full_range/first_div
        second_step = full_range/20/second_div
        best_cut = None
        best_profit = 0

        for i in range(0, first_div+1):

            threshold = parameter_min + first_step*(i-0.5)
            i = searchsorted(full_list[:,0],threshold)
            left, right = full_list[:i], full_list[i:]

            profit_left = left[:,1].sum()
            profit_right = right[:,1].sum()

            left_profit_gain = max(0,(profit_left-root_profit)/max_profit) 
            right_profit_gain = max(0,(profit_right-root_profit)/max_profit)

            if left_profit_gain > best_profit:
                best_cut = threshold
                best_profit = left_profit_gain
            if right_profit_gain > best_profit:
                best_cut = threshold
                best_profit = right_profit_gain

            scores_list.append([threshold, left_profit_gain, right_profit_gain])
    
        if best_cut == None:
            return scores_list    

        parameter_min = best_cut - full_range/40
        if parameter_min < x_parameter.min():
            parameter_min = x_parameter.min()
            
        
        for i in range(0, second_div+1):
            threshold = parameter_min + second_step*(i-0.5)
            i = searchsorted(full_list[:,0],threshold)
            left, right = full_list[:i], full_list[i:]

            profit_left = left[:,1].sum()
            profit_right = right[:,1].sum()

            left_profit_gain = max(0,(profit_left-root_profit)/max_profit) 
            right_profit_gain = max(0,(profit_right-root_profit)/max_profit)

            scores_list.append([threshold, left_profit_gain, right_profit_gain])
            
        return np.array(sorted(scores_list))
    
    def check_best_profit_div(self, divs):
        # Function to return a list [best_div_value, best gain]

        data = np.array(divs)
        left_prediction = 0
        right_prediction = 0

        #Get best profits division

        left_profit_gain = data[:,1].max()
        right_profit_gain = data[:,2].max()
        if left_profit_gain > right_profit_gain:
            profit_gain_list  = data[:,1]
            right_prediction = 1
        else:
            profit_gain_list = data[:,2]
            left_prediction = 1

        best_div = data[np.argmax(profit_gain_list)][0]
        return [best_div, left_prediction, right_prediction, profit_gain_list.max()]


class AdaptiveDecisionTreeClassifier():
    
    def __init__(self,profit_rate, loss_rate, min_samples_split=10, max_depth=6, min_samples_profit=20, min_information_gain = 0, information_gain_cut=0.5):
        ''' constructor '''
        
        # initialize the root of the tree 
        self.root = None
        
        self.profit_rate, self.loss_rate = profit_rate, loss_rate
        
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_samples_profit = min_samples_profit
        self.min_information_gain = min_information_gain
        self.information_gain_cut = information_gain_cut
        
    def build_tree(self, X, y, indices, amounts,  curr_depth=0):
        ''' recursive function to build the tree ''' 
            
        X_ = X.loc[indices]
        y_ = y.loc[indices]
        amounts_ = amounts.loc[indices]
        num_samples = X_.shape[0]
        
        # split until stopping conditions are met
            
        if (num_samples>=self.min_samples_split) & (curr_depth<=self.max_depth):
            # find the best split            
            best_split = self.get_best_split(X_ , y_ , amounts_, num_samples, curr_depth)
            
            # check if information gain is positive
            if best_split["info_gain"]>0:
                left_indices = best_split["dataset_left"]
                right_indices = best_split["dataset_right"]
                # recur left
                left_subtree = self.build_tree(X_, y_, left_indices, amounts_, curr_depth+1)
                # recur right
                right_subtree = self.build_tree(X_, y_, right_indices, amounts_, curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        
        # compute leaf node
        leaf_value = self.calculate_leaf_value(amounts_, y_, self.profit_rate, self.loss_rate)
        # return leaf node
        return Node(value=leaf_value)
    
    def get_best_split(self, X_ , y_ , amounts_, num_samples, curr_depth):
        ''' function to find the best split '''
        
        # dictionary to store the best split
        features = X_.columns
        best_split = {}
        best_percentage = -float("inf")
        
        # Set if the model will calculate profit instead of information gain
        calculate_profit = 0
        
        if (num_samples < self.min_samples_profit) or (curr_depth >= (self.max_depth * self.information_gain_cut)):
            calculate_profit = 1
        
        # loop over all the features
        for feature in features:
            # loop over all the feature values present in the data
            if calculate_profit:
                new_div = self.compute_profit_scores(amounts_, X_[feature], y_)
                div_threshold = self.check_best_profit_div(new_div, curr_depth)
            else:
                new_div = self.compute_information_scores(X_[feature], y_)
                div_threshold = self.check_best_profit_div(new_div, curr_depth)
                    
            if div_threshold[1] > best_percentage:
                best_percentage = div_threshold[1]
                best_variable = feature
                best_div = div_threshold[0]
            
        # cut the data at the best threshold
        dataset_left_index = X_[X_[best_variable] < best_div].index
        dataset_right_index = X_[X_[best_variable] >= best_div].index
            
        # check if childs are not null
        if len(dataset_left_index)>0 and len(dataset_right_index)>0:
            # compute information gain
            best_split["feature_index"] = best_variable
            best_split["threshold"] = best_div
            best_split["dataset_left"] = dataset_left_index
            best_split["dataset_right"] = dataset_right_index
            best_split["info_gain"] = best_percentage
        else:
            best_split["info_gain"] = 0

        # return best split
        return best_split
    
    
    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''
        dataset_left_index = np.where(dataset[:,feature_index]<threshold)
        dataset_right_index = np.where(dataset[:,feature_index]>=threshold)
        return dataset_left_index, dataset_right_index

        
    def calculate_leaf_value(self,amounts_, y_, profit_rate, loss_rate):
        ''' function to compute leaf node '''
        
        profitability = self.check_profitability(amounts_, y_, profit_rate, loss_rate)
        profits = profitability[0] + profitability[1]
        if profits <= 0:
            return 1
        if profits > 0:
            return 0
        
    def compute_entropy(self, y):
        entropy = 0.
        p = np.count_nonzero(y==1)
        if len(y) != 0:
            p = p/len(y)
        else:
            p = 0.5

        if p == 0 or p == 1:
            entropy = 0
        else:
            entropy = -p*np.log2(p) - (1-p)*np.log2(1-p)   
        return entropy
    
    
    def print_tree(self, tree=None, depth=0, lines = None):
        ''' function to print the tree '''
        indent = ""
        if not tree:
            tree = self.root
            
        if not lines:
            lines = []
        line_l = lines.copy()
        line_r = lines.copy()
        line_l.append(1)
        line_r.append(0)
        
        for i in range(depth):
            if lines[i] == 0:
                indent = indent + "    "
            else:
                indent = indent + "│   "

        if tree.value is not None:
            print(tree.value)

        else:
            print(f"{tree.feature_index} < {tree.threshold}  --- Profit gain = {tree.info_gain}")
            print(f"{indent}├───left:", end="")
            self.print_tree(tree.left, depth+1,line_l)
            print(f"{indent}└───right:", end="")
            self.print_tree(tree.right, depth+1,line_r)
    
    def fit(self, X, Y, amounts):
        ''' function to train the tree '''
        
        if len(X) != len(Y):
            print("Length of X and Y don't match")
            return None
        
        indices = X.index
        self.root = self.build_tree(X,Y,indices,amounts)

    
    def make_logic_list(self, X, tree, full_logic_list, leaf_logic_list):
        if tree.value is not None:
            leaf_logic_list.append(tree.value)
        else:
            leaf_logic_list_right = leaf_logic_list.copy()
            
            leaf_logic_list.append((X[tree.feature_index] < tree.threshold))
            leaf_logic_list_right.append((X[tree.feature_index] >= tree.threshold))
            
            full_logic_list.append(leaf_logic_list_right)
            
            self.make_logic_list( X, tree.left, full_logic_list, leaf_logic_list)
            self.make_logic_list( X, tree.right, full_logic_list, leaf_logic_list_right)
    
    def predict(self, X):
        ''' function to predict new dataset '''
        
        tree = self.root
        full_logic_list = []
        leaf_logic_list= []
        full_logic_list.append(leaf_logic_list)
        self.make_logic_list(X, tree, full_logic_list, leaf_logic_list)
        prediction = pd.Series(0, index=X.index)
        
        for leaf_logics in full_logic_list:
            new_logic = leaf_logics[0]
            for logic in leaf_logics[1:-1]:
                new_logic = new_logic & logic
            prediction[new_logic] = leaf_logics[-1]
        return prediction
    
    def check_profitability(self, x_transactions, y_data, profit_rate, loss_rate):
        # Gets the transactions values and labels and returns percentage of profitability and profitability amount
    
        profits = x_transactions.loc[y_data[y_data == 0].index].sum() * profit_rate
        losses = x_transactions.loc[y_data[y_data == 1].index].sum() * loss_rate * -1
    
        return [profits, losses]
    
    
    def compute_profit_scores(self, x_transactions, x_parameter, y_sample):  
        scores_list = []

        final_value = x_transactions.copy()
        final_value[y_sample==0] = final_value[y_sample==0]*self.profit_rate
        final_value[y_sample==1] = final_value[y_sample==1]*self.loss_rate*-1
        max_profit = final_value[y_sample==0].sum()
        root_profit = final_value.sum()
        
        full_list = np.stack((x_parameter, final_value), axis=1)
        full_list = full_list[full_list[:, 0].argsort()]
        
        parameter_min = x_parameter.min()
        parameter_max = x_parameter.max()
        full_range = parameter_max - parameter_min
        
        first_div = 40
        second_div = 30
        third_div = 20
        
        first_step = full_range/first_div
        second_step = full_range/10/second_div
        third_step = full_range/100/third_div

        best_profit_cut = None
        best_profit_gain = 0
        profitable_side = None
        
        if max_profit == 0:
            return [[0,0]]
        
        if root_profit<=0:
            root_profit = 0
        
        # First check
        for i in range(0, first_div+1):

            threshold = parameter_min + first_step*(i-0.5)
            i = searchsorted(full_list[:,0],threshold)
            left, right = full_list[:i], full_list[i:]
            profit_left = left[:,1].sum()
            profit_right = right[:,1].sum()

            # Get profits gain
            left_profit_gain = max(0,(profit_left-root_profit)/max_profit) 
            right_profit_gain = max(0,(profit_right-root_profit)/max_profit)
            
            gain = left_profit_gain
            if right_profit_gain > left_profit_gain:
                gain = right_profit_gain
            if left_profit_gain > best_profit_gain:
                best_profit_cut = threshold
                best_profit_gain = left_profit_gain
                side = "Left"
            if right_profit_gain > best_profit_gain:
                best_profit_cut = threshold
                best_profit_gain = right_profit_gain
                side = "Right"

            scores_list.append([threshold, gain])
        
        if (best_profit_cut == None):
            return scores_list
        
        parameter_min = best_profit_cut - full_range/20
        
        for i in range(0, second_div+1):
            threshold = parameter_min + second_step*(i-0.5)
            i = searchsorted(full_list[:,0],threshold)
            if side == "Left":
                profit_list = full_list[:i]
            else:
                profit_list = full_list[i:]
            profit = profit_list[:,1].sum()

            # Get profits gain
            profit_gain = max(0,(profit-root_profit)/max_profit) 
            
            
            if profit_gain > best_profit_gain:
                best_profit_cut = threshold
                best_profit_gain = profit_gain

            scores_list.append([threshold, gain])
        
        parameter_min = best_profit_cut - full_range/200
        
        for i in range(0, third_div+1):
            threshold = parameter_min + third_step*(i-0.5)
            i = searchsorted(full_list[:,0],threshold)
            if side == "Left":
                profit_list = full_list[:i]
            else:
                profit_list = full_list[i:]
            profit = profit_list[:,1].sum()

            # Get profits gain
            profit_gain = max(0,(profit-root_profit)/max_profit) 
            
            
            if profit_gain > best_profit_gain:
                best_profit_cut = threshold
                best_profit_gain = profit_gain

            scores_list.append([threshold, gain])
            
        return np.array(sorted(scores_list))
    
    def compute_information_scores(self, x_parameter, y_sample):  
        scores_list = []
        parameter_min = x_parameter.min()
        parameter_max = x_parameter.max()

        full_list = np.stack((x_parameter, y_sample), axis=1)
        full_list = full_list[full_list[:, 0].argsort()]

        full_range = parameter_max - parameter_min
        first_div = 40
        second_div = 30
        third_div = 20
        
        
        first_step = full_range/first_div
        second_step = full_range/10/second_div
        third_step = full_range/100/third_div

        best_information_cut = None
        best_information_gain = 0
        
        n_root = len(y_sample)
        root_entropy = self.compute_entropy(full_list[:,1])
        information_gain = 0
        
        # First check
        for i in range(0, first_div+1):
            threshold = parameter_min + first_step*(i-0.5)
            i = searchsorted(full_list[:,0],threshold)
            left, right = full_list[:i], full_list[i:]
            n_left = i
            n_right = n_root-i
            left_entropy = self.compute_entropy(left[:,1])
            right_entropy = self.compute_entropy(right[:,1])
            
            left_weight = n_left/n_root
            right_weight = n_right/n_root
            
            information_gain = root_entropy - (left_entropy*left_weight + right_entropy*right_weight)

            if information_gain > best_information_gain:
                best_information_cut = threshold
                best_information_gain = information_gain
                
            scores_list.append([threshold, information_gain])
        
        
        # Second check
        # If the first check had a low information gain, return None
        if (best_information_gain <= self.min_information_gain):
            return scores_list
        
        # Set the start of the next check
        parameter_min = best_information_cut - (full_range/20)
        if parameter_min < x_parameter.min():
            parameter_min = x_parameter.min()
        
        for i in range(0, second_div+1):
            threshold = parameter_min + second_step*(i-0.5)
            i = searchsorted(full_list[:,0],threshold)
            left, right = full_list[:i], full_list[i:]
            n_left = i
            n_right = n_root-i
            left_entropy = self.compute_entropy(left[:,1])
            right_entropy = self.compute_entropy(right[:,1])
            
            left_weight = n_left/n_root
            right_weight = n_right/n_root
            
            information_gain = root_entropy - (left_entropy*left_weight + right_entropy*right_weight)

            if information_gain > best_information_gain:
                best_information_cut = threshold
                best_information_gain = information_gain
                
            scores_list.append([threshold, information_gain])
        
        # Set the start of the next check
        parameter_min = best_information_cut - (full_range/200)
        if parameter_min < x_parameter.min():
            parameter_min = x_parameter.min()
        
        for i in range(0, third_div+1):
            threshold = parameter_min + third_step*(i-0.5)
            i = searchsorted(full_list[:,0],threshold)
            left, right = full_list[:i], full_list[i:]
            n_left = i
            n_right = n_root-i
            left_entropy = self.compute_entropy(left[:,1])
            right_entropy = self.compute_entropy(right[:,1])
            
            left_weight = n_left/n_root
            right_weight = n_right/n_root
            
            information_gain = root_entropy - (left_entropy*left_weight + right_entropy*right_weight)

            if information_gain > best_information_gain:
                best_information_cut = threshold
                best_information_gain = information_gain
                
            scores_list.append([threshold, information_gain])

        return np.array(sorted(scores_list))
    
    def check_best_profit_div(self, divs, curr_depth):
        # Function to return a list [best_div_value, left_prediction, right_prediction, best gain]

        data = np.array(divs)
        gain_list = data[:,1]
        best_div = data[np.argmax(gain_list)][0]
        return [best_div, gain_list.max()]
    

class CustomRandomForestClassifier():
    def __init__(self, min_samples_split=10, max_depth=4, num_trees=100, columns_per_tree = None, information_gain_cut=0.3, min_samples_profit=50):
        ''' constructor '''
        
        # initialize an empty tree to store the trees 
        self.trees = []
        
        # stopping conditions
        self.columns_per_tree = columns_per_tree
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.num_trees = num_trees
        self.information_gain_cut = information_gain_cut
        self.min_samples_profit = min_samples_profit
    
    def fit(self, X, Y, amounts, profit_rate, loss_rate):
        ''' function to train the tree '''
        if not self.columns_per_tree:
            columns_per_tree = int(math.sqrt(len(X.columns)))
        else:
            columns_per_tree = self.columns_per_tree
        all_features = X.columns
        len_0 = len(Y[Y==0])
        len_1 = len(Y[Y==1])
        X_0 = X[Y==0]
        X_1 = X[Y==1]
        
        for i in range(self.num_trees):
            sel_features = random.sample(list(all_features), columns_per_tree)
            X_ = X[sel_features].sample(int(X.shape[0]/2), replace= True)

            # classifier = ProfitDecisionTreeClassifier(profit_rate, loss_rate, min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            classifier = AdaptiveDecisionTreeClassifier(
                profit_rate, loss_rate, min_samples_split=self.min_samples_split, max_depth=self.max_depth,
                min_samples_profit=self.min_samples_profit, information_gain_cut=self.information_gain_cut)
            
            classifier.fit(X,Y,amounts)
            self.trees.append(classifier)
            # classifier.print_tree()
            if i%10 == 0:
                print(f"Trained {i} trees") 
        
        print("Finished training")
        # self.root = self.build_tree(X,y,indices,amounts, profit_rate, loss_rate)
    
    def predict_proba(self, X):
        ''' function to predict new dataset '''
        prediction_list = pd.DataFrame(index=X.index)
        classifiers = self.trees
        num_classifiers = len(classifiers)
        for i in range(num_classifiers):
            prediction_list[i] =  classifiers[i].predict(X)
        return prediction_list.mean(axis=1)
    
    def predict(self, X, threshold=0.5):
        
        predictions_prob = self.predict_proba(X)
        predictions = predictions_prob.apply(lambda s: 1 if s>=threshold else 0)
        
        return predictions