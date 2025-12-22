# :wrench: Decision Tree From Scratch :hammer:

A from-scratch implementation of a Decision Tree classifier in Python, built to understand the internal mechanics of tree-based machine learning models.

---

## Motivation & Learning Goals

The goal of this project is educational.

By implementing a Decision Tree from scratch, I aim to:

- Understand how tree-based models work internally
- Learn how impurity measures guide split decisions
- Practice recursive algorithms and clean code design
- Identify the limitations of naive implementations

This project intentionally avoids using libraries such as `scikit-learn` for the core learning algorithm.

---

## High-Level Overview

At a high level, this implementation follows these steps:

1. Start at the root node with the full dataset
2. Evaluate all possible splits across all features
3. Select the split that produces the highest impurity reduction
4. Recursively repeat the process for each child node
5. Stop when a stopping condition is met and create a leaf node

Predictions are made by traversing the tree from the root to a leaf node based on feature values.

---

## Features Implemented

- Binary splits on numerical features
- Impurity-based split selection (e.g. Gini impurity or entropy)
- Recursive tree construction
- Configurable stopping criteria (e.g. maximum depth)
- Prediction on unseen samples

---

## Project Structure

    decision-tree/
    │── decision_tree.py    # Core Decision Tree implementation
    │── README.md           # Project documentation

- `decision_tree.py`: Contains the full implementation of the Decision Tree, including node representation, tree construction, and prediction logic.

---

## Key Design Decisions

- The tree is built recursively to mirror the theoretical algorithm
- Nodes are represented explicitly to distinguish decision nodes from leaf nodes
- Splitting logic is kept modular to make impurity measures easier to extend

These decisions prioritize clarity and learning over performance.

---

## Limitations & Known Issues

This implementation is intentionally minimal and has several limitations:

- No pruning (pre-pruning or post-pruning)
- Only supports classification
- No handling of missing values
- No categorical feature support
- Not optimized for large datasets

These limitations are documented as part of the learning process.

---

## Comparison with scikit-learn

Compared to `sklearn.tree.DecisionTreeClassifier`, this implementation:

- Focuses on transparency rather than performance
- Omits advanced optimizations and pruning strategies
- Uses simpler data structures
- Is not intended for production use

The goal is conceptual understanding, not feature parity.

---

## Usage

Example usage:

    from decision_tree import DecisionTree

    model = DecisionTree(max_depth=3)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

Input data is expected to be numerical and provided as NumPy arrays or equivalent structures.

---

## Future Improvements

Potential future extensions include:

- Implementing pruning techniques
- Supporting regression trees
- Adding feature importance computation
- Improving computational efficiency
- Visualizing the tree structure

---

## References

- CART algorithm
- Information gain and Gini impurity

---
