#include "id3regressor.h"

#include <random>

// Generate a simple artificial dataset
std::tuple<std::vector<std::vector<double>>, std::vector<double>> generate_dataset(int n_samples) {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-10, 10);
    
    for (int i = 0; i < n_samples; i++) {
        double x1 = dis(gen);  // Feature 1
        double x2 = dis(gen);  // Feature 2
        
        // Create a simple linear relation for the target variable
        // y = 3 * x1 + 2 * x2 + noise
        double noise = dis(gen) * 0.5;
        double target = 3 * x1 + 2 * x2 + noise;
        
        X.push_back({x1, x2});
        y.push_back(target);
    }
    
    return std::make_tuple(X, y);
}

int main() {
    // Generate the dataset
    auto [X, y] = generate_dataset(100);
    
    // Print the dataset to verify
    std::cout << "Feature 1\tFeature 2\tTarget\n";
    for (int i = 0; i < X.size(); i++) {
        std::cout << X[i][0] << "\t    " << X[i][1] << "\t    " << y[i] << "\n";
    }

    // Initialize and fit the ID3 regressor model
    ID3Regressor model;
    model.fit(X, y);
    
    // Predict on the same dataset and print predictions
    std::vector<double> predictions = model.predict(X);
    std::cout << "\nPredictions:\n";
    for (const auto& pred : predictions) {
        std::cout << pred << "\n";
    }

    return 0;
}