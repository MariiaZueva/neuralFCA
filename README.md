# Neural FCA

A big homework to merge [Neural Networks](https://en.wikipedia.org/wiki/Artificial_neural_network)
and [Formal Concept Analysis](https://en.wikipedia.org/wiki/Formal_concept_analysis).
Created for the course Ordered Sets in Data Analysis (GitHub [repo](https://github.com/EgorDudyrev/OSDA_course))
taught in Data Science Master programme in HSE University, Moscow. 

![Example of a network build upon Fruits dataset](https://github.com/EgorDudyrev/OSDA_course/blob/Autumn_2022/neural_fca/fitted_network.png)

# Description
There is a need for interpretability in the AI field. It is usually hard to interpret the performance of neural networks. But there is an approach to create an interpretable neural network architecture based on the covering relation (graph of the diagram) of a lattice coming from monotone Galois connections. The vertices of such neural networks are related to sets of similar objects with similarity given by their common attributes, so easily interpretable. The edges between vertices are also easily interpretable in terms of concept generality (bottom-up) or conditional probability (top-bottom).


# OSDA 2023 Neural FCA

This repository contains tools for a "*Neural FCA**" big homework assignment for the course "Ordered Sets in Data Analysis" taught at HSE in the "Data Science" master's program in Fall 2023.


### To-do list

1. Choose at least 3 datasets (Kaggle, UCI, etc.), define the target attribute, binarize data and describe scaling (binarization) strategy for the dataset features.
(Ideal dataset: openly available, with various data types with hundreds of rows.)\
Useful resources:
* [Kaggle](https://www.kaggle.com/)
* [UCI repository](https://archive.ics.uci.edu/datasets)
2. Perform classification using standard ML tools:
* [Decision tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) 
* [Random forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier)
* [xGboost](https://xgboost.readthedocs.io/en/latest/)
* [Catboost](https://catboost.ai/)
* [k-NN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
* [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
* [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)
3. Describe prediction quality measure best suited for the dataset (e.g. accuracy, F1 score, or any quality measure best suited for the dataset; fit and test the network on your task.
4. Try  to improve the basic baseline with different ways such that:
  - better scaling algorithm to binarize the original data;
  - use various techniques to select best concepts from the concept lattice;
  - try various nonlinearities to put in the network.
5.  Submit a report with comparison of all models, both standard and developed by you.
