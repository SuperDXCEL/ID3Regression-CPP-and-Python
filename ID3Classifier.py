class ID3Classifier:
    def __init__(self):
        self.tree = None

    class Node:
        def __init__(self, feature_index=None, threshold=None, value=None, left=None, right=None):
            self.feature_index = feature_index  # Index of feature to split on
            self.threshold = threshold  # Threshold value for the split
            self.value = value  # Predicted class (for leaf nodes)
            self.left = left  # Left child node
            self.right = right  # Right child node

    def entropy(self, target_variable):
        # Get the counts of each class
        _, class_counts = np.unique(target_variable, return_counts=True)
        # Probability of each class
        class_probabilities = class_counts / len(target_variable)
        # Calculate entropy
        return -np.sum(class_probabilities * np.log2(class_probabilities + 1e-10))

    def split(self, X, feature, threshold):
        # Split the dataset based on the feature and threshold
        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold
        return left_mask, right_mask

    def information_gain(self, X, y, feature, threshold):
        # Split the dataset based on the feature and threshold
        left_mask, right_mask = self.split(X, feature, threshold)
        left_y, right_y = y[left_mask], y[right_mask]

        # Calculate entropy of the parent node
        parent_entropy = self.entropy(y)
        
        # Calculate the entropy of the left and right child nodes
        left_entropy = self.entropy(left_y)
        right_entropy = self.entropy(right_y)
        
        # Calculate the weighted average entropy of the children
        left_weight = len(left_y) / len(y)
        right_weight = len(right_y) / len(y)
        weighted_entropy = left_weight * left_entropy + right_weight * right_entropy
        
        # Return information gain
        return parent_entropy - weighted_entropy

    def find_best_split(self, X, y):
        num_features = X.shape[1]
        best_feature, best_threshold, max_gain = None, None, -1

        for feature in range(num_features):
            # Get unique values in the feature column (thresholds)
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self.information_gain(X, y, feature, threshold)
                if gain > max_gain:
                    max_gain = gain
                    best_threshold = threshold
                    best_feature = feature
        
        return best_feature, best_threshold

    def build_tree(self, X, y, depth=0, max_depth=None):
        # If max depth is reached or all the target variables have the same class
        if depth == max_depth or len(set(y)) == 1:
            return self.Node(value=np.argmax(np.bincount(y)))

        # Find the best feature and threshold to split on
        feature, threshold = self.find_best_split(X, y)
        
        # If no split is possible, create a leaf node
        if feature is None:
            return self.Node(value=np.argmax(np.bincount(y)))

        # Split the dataset
        left_mask, right_mask = self.split(X, feature, threshold)
        left_X, left_y = X[left_mask], y[left_mask]
        right_X, right_y = X[right_mask], y[right_mask]

        # Recursively build left and right subtrees
        left = self.build_tree(left_X, left_y, depth + 1, max_depth)
        right = self.build_tree(right_X, right_y, depth + 1, max_depth)

        # Return the node containing the feature, threshold, and subtrees
        return self.Node(feature_index=feature, threshold=threshold, left=left, right=right)

    def fit(self, X, y, max_depth=None):
        # Build the decision tree
        self.tree = self.build_tree(X, y, max_depth)

    def predict_one(self, node, sample):
        # Predict a single sample by traversing the decision tree
        if node.value is not None:
            return node.value
        if sample[node.feature_index] <= node.threshold:
            return self.predict_one(node.left, sample)
        else:
            return self.predict_one(node.right, sample)

    def predict(self, X):
        # Predict all samples in the dataset
        return [self.predict_one(self.tree, sample) for sample in X]
