# CS771-Guess_the_word
This is a course project for the course CS771: intro to machine learning (IIT Kanpur).

## Overview
The game involves Melbo guessing a secret word chosen from a dictionary, with Melbot providing feedback in the form of character matches. The goal is to create an efficient decision tree algorithm to maximize the number of correctly guessed words within the query limit.

## Problem Description

The game proceeds as follows:
1. Melbot chooses a secret word from a known dictionary.
2. Melbot informs Melbo of the word length by sending a string of underscores.
3. Melbo makes guesses by providing indices corresponding to words in the dictionary.
4. Melbot provides feedback on the guessed word and may terminate the round if the guess is correct or if the number of queries exceeds the limit.

## Solution

### Decision Tree Algorithm

1. **Decision Tree Structure**:
   - **Root Node**: Contains all words in the dictionary.
   - **Leaf Nodes**: Each node corresponds to a single word from the dictionary or a node where the maximum depth (15) is reached.

2. **Splitting Criterion**:
   - Each node queries the word that provides the maximum information gain. Information gain is calculated based on the entropy of the words at the node and the entropy of the resulting child nodes.

3. **Node Expansion**:
   - Nodes are expanded until they contain a single word or reach the maximum depth of 15.
   - Words are distributed among child nodes based on the feedback from Melbot (e.g., characters in the correct positions).

4. **Improvements**:
   - Reduced training time by iterating over a subset of words rather than all words at each node.
   - For nodes with more than 500 words, 10% of words are sampled; for nodes with 50-500 words, 50 words are sampled; for smaller nodes, all words are checked.

5. **Gameplay**:
   - The decision tree is used to make guesses and update the current node based on Melbot's feedback.
   - If the guess matches the secret word, the round ends and the win count is incremented.

### Files

- **`Decition_tree_code.py`**: Contains the implementation of the decision tree learning algorithm. The file includes:
  - `my_fit()` method: Trains the decision tree model on the given dictionary.
  - Tree and Node classes for managing the decision tree structure.

- **`template.py`**: Provides a template for submission and helps in validating the output.
- **`data files`**: the dictionary the model was trained on. Train and test files are also provided.
- **`report.pdf`**: Detailed explanation of the design decisions, including the splitting criterion, stopping conditions, pruning strategies, and hyperparameters used.

