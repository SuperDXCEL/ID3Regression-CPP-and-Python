#include "id3regressor.h"

std::tuple<std::vector<double>, std::vector<double>> split_target(std::vector<double> target_list, double threshold) {
  std::vector<double> left, right;
  for (int i = 0; i < target_list.size(); i++) {
    target_list[i] <= threshold ? left.push_back(target_list[i]) : right.push_back(target_list[i]);  
  }
  return make_tuple(left, right);
}
double get_mean_value(const std::vector<double>& values) {
  double mean = 0;
  for (int i = 0; i < values.size(); i++) {
    mean += values[i];
  }
  return mean / values.size();  
}
void get_unique(std::vector<double>& values) {
  // Get unique values in the vector
  std::sort(values.begin(), values.end());
  auto last = std::unique(values.begin(), values.end());
  values.erase(last, values.end());
}
double ID3Regressor::entropy(std::vector<double> target_variable) {
  // Get unique values in the vector
  get_unique(target_variable);
  // Get the probabilities of each class in the target_variable
  std::vector<double> class_probabilities;
  double sumatory = 0;
  for (int i = 0; i < target_variable.size(); i++) {
    class_probabilities.push_back(target_variable[i] / target_variable.size());
    sumatory += class_probabilities[i] * log2(class_probabilities[i] + 1e-10);
  }
  return -sumatory;
}
std::tuple <std::vector<std::vector<double>>, std::vector<std::vector<double>>> ID3Regressor::split(std::vector<std::vector<double>> X, int feature_index, double threshold) {
  std::vector<std::vector<double>> left, right;
  for (int i = 0; i < X.size(); i++) {
    X[i][feature_index] <= threshold ? left.push_back(X[i]) : right.push_back(X[i]);
  }
  // std::cout << "SIZES: " << left.size() << ", " << right.size() << std::endl;
  return std::make_tuple(left, right);
}
double ID3Regressor::information_gain(std::vector<std::vector<double>> X, std::vector<double> target_variable, int feature_index, double threshold) {
  // Split the dataset and the target variable
  auto [left_target, right_target] = split_target(target_variable, threshold);  // Use split_target for target
  // Calculate entropies
  double left_entropy = entropy(left_target);
  double right_entropy = entropy(right_target);
  // Calculate weights
  double left_weight = static_cast<double>(left_target.size() / target_variable.size());
  double right_weight = static_cast<double>(right_target.size() / target_variable.size());
  // Calculate information gain
  double information_gain_value = entropy(target_variable) - (left_weight * left_entropy + right_weight * right_entropy);
  return information_gain_value;
}
std::tuple <int, double> ID3Regressor::find_best_split(std::vector<std::vector<double>> X, std::vector<double> target_variable) {
  double best_threshold{-1}, best_feature{-1}, max_gain{0};
  for (int i = 0; i < X[0].size(); i++) {
    std::vector<double> thresholds;
    for (int j = 0; j < X.size(); j++) {
      thresholds.push_back(X[j][i]);
    }
    get_unique(thresholds);
    for (const auto& threshold : thresholds) {
      double gain = information_gain(X, target_variable, i, threshold);
      if (gain > max_gain) {
        max_gain = gain;
        best_feature = i;
        best_threshold = threshold;
      }
    }
  }
  return std::make_tuple(best_feature, best_threshold);
}
Node* ID3Regressor::build_tree(std::vector<std::vector<double>> X, std::vector<double> target_variable, int depth=0, int max_depth) {
  std::unordered_set<double> target_variable_set(target_variable.begin(), target_variable.end());
  // Base case where size == 1 or max_depth has been reached
  if (depth == max_depth || target_variable_set.size() == 1) {
    double value = get_mean_value(target_variable);
    return new Node(-1, -1, value, nullptr, nullptr);
  }
  // Find the best split
  auto [best_feature, best_threshold] = find_best_split(X, target_variable);
  if (best_feature == -1) {
    double value = get_mean_value(target_variable);
    return new Node(-1, -1, value, nullptr, nullptr);
  }
  // Split the dataset
  auto [left_dataset, right_dataset] = split(X, best_feature, best_threshold);
  // Split the target variable according to the dataset split
  std::vector<double> left_target, right_target;
  for (int i = 0; i < X.size(); i++) {
    if (X[i][best_feature] < best_threshold) {
      left_target.push_back(target_variable[i]);
    } else {
      right_target.push_back(target_variable[i]);
    }
  }
  // Recursively build the tree
  Node* left_node = build_tree(left_dataset, left_target, depth+1, max_depth);
  Node* right_node = build_tree(right_dataset, right_target, depth+1, max_depth);
  // Return the top node
  return new Node(best_feature, best_threshold, -1, left_node, right_node);
}
void ID3Regressor::fit(std::vector<std::vector<double>> X, std::vector<double> target_variable, int max_depth) {
  initial_node_ = build_tree(X, target_variable, max_depth);
}
double ID3Regressor::predict_one(Node* node, std::vector<double> sample) {
  if (node->left_ == nullptr && node->right_ == nullptr) {
    return node->value_;
  }
  if (sample[node->feature_index_] <= node->threshold_) {
    return predict_one(node->left_, sample);
  }
  else {
    return predict_one(node->right_, sample);
  }
}
std::vector<double> ID3Regressor::predict(std::vector<std::vector<double>> X) {
  Node* init = initial_node_;
  std::vector<double> predictions;
  for (const auto& row : X) {
    double prediction = predict_one(init, row);
    predictions.push_back(prediction);
  }
  return predictions;
}