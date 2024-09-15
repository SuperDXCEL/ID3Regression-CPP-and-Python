#ifndef ID3REGRESSOR_H
#define ID3REGRESSOR_H

#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <tuple>
#include <optional>
#include <unordered_map>
#include <unordered_set>

class Node {
friend class ID3Regressor;
public:
  Node(int feature_index=-1, double threshold=-1, double value=-1, Node* left=nullptr, Node* right=nullptr)
      : feature_index_(feature_index), threshold_(threshold), value_(value), left_(left), right_(right) {}
private:
  int feature_index_;
  double threshold_;
  double value_;
  Node* left_;
  Node* right_;
};

class ID3Regressor {
public:
  ID3Regressor() {}
  // Get the entropy (related to surprise) of the target_variable
  double entropy(std::vector<double> target_variable);
  // Splitting the data into smaller (left) and bigger (right) than the threshold
  std::tuple <std::vector<std::vector<double>>, std::vector<std::vector<double>>> split(std::vector<std::vector<double>> X, int feature_index, double threshold);
  // Find the information gain, to find what feature and where in that feature to split
  double information_gain(std::vector<std::vector<double>> X, std::vector<double> target_variable, int feature_index, double threshold);
  // Find the best split possible
  std::tuple <int, double> find_best_split(std::vector<std::vector<double>> X, std::vector<double> target_variable);
  // Recursively build the tree
  Node* build_tree(std::vector<std::vector<double>> X, std::vector<double> target_variable, int depth, int max_depth=5);
  // Fit the data to the tree
  void fit(std::vector<std::vector<double>> X, std::vector<double> target_variable, int max_depth=5);
  // Predict one data point
  double predict_one(Node* node, std::vector<double> sample);
  // Predict all data points
  std::vector<double> predict(std::vector<std::vector<double>> X);
private:
  Node* initial_node_;
};

#endif // ID3REGRESSOR_H