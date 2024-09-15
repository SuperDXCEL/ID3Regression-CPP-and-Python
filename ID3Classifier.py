import numpy as np

class ID3Classifier:
    def __init__(self):
        self.tree = None

    class Node:
        def __init__(self, feature_index=None, threshold=None, value=None, left=None, right=None):
            self.feature_index = feature_index # Index of feature to split on
            self.threshold = threshold # Threshold value for the split
            self.value = value # Predicted class (for leaf nodes)
            self.left = left # Left child node
            self.right = right # Right child node
    
    def entropy(self, target_variable):
        # Get the counts of each class
        _, class_counts = np.unique(target_variable, return_counts=True)
        # Probability of each class by dividing each target count by the length of the target_variable 
        class_probabilities = class_counts / len(target_variable)
        # np.sum iterates through the array and gets the target_variable entropy
        return -np.sum(class_probabilities * np.log2(class_probabilities + 1e-10))
    
    def split(X, feature, threshold):
        # Split the dataset based on the feature and the threshold
        left = X[X[:, feature] <= threshold] 
        right = X[X[:, feature] > threshold]
        return left, right

    def information_gain(self, X, target_variable, feature, threshold):
        # Information Gain = Entropy of parent node − (weighted entropy of left child + weighted entropy of right child)
        left, right = self.split(X, feature, threshold)
        left_entropy = self.entropy(left[:, target_variable])
        right_entropy = self.entropy(right[:, target_variable])
        left_weight = len(left) / len(X)
        right_weight = len(right) / len(X) 
        return self.entropy(X[:, target_variable]) - (left_weight * left_entropy + right_weight * right_entropy)
    
    def find_best_split(self, X, target_variable):
        num_features = X.shape[1]
        best_feature, best_threshold, max_gain = None, None, -1
        for feature in range(num_features):
            # Get all unique values (possible thresholds) in the feature column
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self.information_gain(X, target_variable, feature, threshold)
                if (gain > max_gain):
                    max_gain = gain
                    best_threshold = threshold
                    best_feature = feature
        return best_feature, best_threshold
    
    def build_tree(self, X, target_variable, depth=0, max_depth=None):
        if depth == max_depth or len(set(y)) == 1:
            return self.Node(value=np.argmax(np.bincount(y)))
        
        feature, threshold = self.find_best_split(X, target_variable)
        if feature is None:
            return self.Node(value=np.argmax(np.bincount(y)))

        left, right = split(X, feature, threshold)
        left = self.build_tree(X[left], target_variable[left], depth + 1, max_depth)
        right = self.build_tree(X[right], target_variable[right], depth + 1, max_depth)

        return self.Node(feature_index=feature, threshold=threshold, left=left, right=right)
    
    def fit(self, X, y, max_depth=None):
        self.tree = self.build_tree(X, y, max_depth=None)

    def predict_one(self, node, sample):
        if node.value is not None:
            return node.value
        if sample[node.feature] <= node.threshold:
            return self.predict_one(node.left, sample)
        else:
            return self.predict_one(node.right, sample)
    
    def predict(self, X):
        return [self.predict_one(self.tree, sample) for sample in X]