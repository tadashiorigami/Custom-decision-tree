# Custom decision tree

Decision trees from scratch with custom criterion.

This is a custom library with two decision trees algorithm I made in Python to help me better understand the workings of the model.

There are two decision tree classifiers in the library:

* ProfitsDecisionTreeClassifier()
* AdaptiveDecisionTreeClassifier()

The decision trees are made specifically for credits defaults and chargebacks analisys. Instead of making decisions based on GINI or Entropy, the decision trees uses "profits" as the criterion for each classification.

The ProfitsDecisionTreeClassifier() uses profits as the criterion for every node.

The AdaptiveDecisionTreeClassifier() uses entropy for the initial nodes and profits on the other nodes, based on the information gain and profit gain.

There's also a third model called "AdaptiveRandomForestClassifier()" still in deveopment, but a very crude first example is also available.

The jupyter notebook shows the results of a few tests on a kaggle dataset.

## Training the ProfitDecisionTreeClassifier() model

To train the model, pass the profit and loss rate (0~1) of each transaction and use model.fit passing the dataframes X, y and the value of each transaction to calculate the profit and losses.

model = ProfitDecisionTreeClassifier(profit_rate, loss_rate)

model.fit(X,y,transactions)

You can also pass the parameters: 

- min_samples_split : The minimum amount of samples in a node to divide it further. 
- max_depth : Max depth of the decision tree

Lastly, you can also print the tree with model.print_tree()

## Training the AdaptiveDecisionTreeClassifier() model

The adaptive decision tree classifier works the same way as the profits decision tree, but you can also pass other parameters:

- min_samples_profit : Minimum samples in a node to calculate information gain. If minimum is not met, the criteria will be changed to "profit gain" (Yes, I know, I will change the name)
- min_information_gain : Minimum information gain needed in a node. If the minimum is not met, the criteria will be changed to "profit gain"
- information_gain_cut : Value between 0 and 1. Percentage of the max depth where "information gain" criteria will be used before changing it to "profit gain"
