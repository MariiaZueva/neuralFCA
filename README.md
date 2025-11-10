# Neural FCA

A big homework to merge neural networks
and Formal Concept Analysis.

# Description
There is a need for interpretability in the AI field. It is usually hard to interpret the performance of neural networks. But there is an approach to create an interpretable neural network architecture based on the covering relation (graph of the diagram) of a lattice coming from monotone Galois connections. The vertices of such neural networks are related to sets of similar objects with similarity given by their common attributes, so easily interpretable. The edges between vertices are also easily interpretable in terms of concept generality (bottom-up) or conditional probability (top-bottom).


# OSDA 2025 Neural FCA

This repository contains tools for a "**Neural FCA**" big homework assignment for the course "Ordered Sets in Data Analysis" taught at HSE in the "Data Science" master's program in Fall 2023.


### To-do list

1. Choose at least 3 datasets (Kaggle, UCI, etc.), define the target attribute, binarize data and describe scaling (binarization) strategy for the dataset features.
(Ideal dataset: openly available, with various data types with hundreds of rows.)\

2. Perform classification using standard ML tools (at least 4), explain your choice. For example:
- K Nearest Neighbor (kNN)
- Naive Bayes
- Logistic Regression
- SVM
- Decision Tree
- Random Forest
- XGBoost

3. Describe prediction quality measure best suited for the dataset (e.g. accuracy, F1 score, or any quality measure best suited for the dataset; fit and test the network on your task.
   
4. Try  to improve the basic baseline with different ways such that:
  - better scaling algorithm to binarize the original data;
  - use various techniques to select best concepts from the concept lattice;
  - propose your own modification of this nn's architecture.
5.  Submit a report with comparison of all models, both standard and developed by you. Analyse your results.


Grading system (out of 10):
- Performing steps 1-6 will grants a maximum of 3/10.
- Improving on the algorithm grants a maximum of 2/10.
- Intepretability analysis grants a maximum of 2/10.
- Presenting the work grants a maximum of 3/10.
- (p.s.!!!) Not uploading the report before the deadline will result in a 0 mark.


### References
1. Kuznetsov, Sergei O., Nurtas Makhazhanov, and Maxim Ushakov. "On neural network architecture based on concept lattices." Foundations of Intelligent Systems: 23rd International Symposium, ISMIS 2017, Warsaw, Poland, June 26-29, 2017, Proceedings 23. Springer International Publishing, 2017.
