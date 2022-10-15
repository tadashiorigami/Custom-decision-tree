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
