### 1. **Training Set:**

- Ideally, you want a training set that represents each age group and both types of connectivity (SC and FC). Since you have 30 graphs total, you should allocate a majority of these for training.
- **Recommendation:** Use approximately 60-70% of your total dataset for training.
- For 30 graphs, that would be about 18 to 21 graphs.

### 2. **Validation Set:**

- Use a portion of the graphs to validate the model during training. This helps in tuning hyperparameters and avoiding overfitting.
- **Recommendation:** Use about 10-20% of your total dataset for validation.
- For 30 graphs, that would be about 3 to 6 graphs.

### 3. **Test Set:**

- The remaining graphs should be reserved for testing the model's performance after training.
- **Recommendation:** Use about 20-30% of your total dataset for testing.
- For 30 graphs, that would be about 6 to 9 graphs.

### **Summary:**

- **Training Set:** 18 to 21 graphs
- **Validation Set:** 3 to 6 graphs
- **Test Set:** 6 to 9 graphs

### **Considerations:**

- Ensure that the training set includes a balanced representation of SC and FC graphs from each age group.
- If possible, ==apply data augmentation techniques to increase the diversity of the training set without compromising the test set==.
- To maximize the effectiveness of your GNN, ==consider cross-validation, where you rotate the training, validation, and test sets across different runs to make use of all 30 graphs==.

By using approximately 18 to 21 graphs for training, you can build a robust model while still leaving sufficient data for validation and testing.