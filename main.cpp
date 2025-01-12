#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <random>
#include <iomanip>
#include <numeric>
#include <fstream>
#include <chrono>
#include <sstream>
#include <unordered_set>

/* Code *can* implement IBCF, but implements NCF and SVD as a hybrid recommender for better speed. */

class HybridRecommender {
private:
    // SVD components
    std::vector<std::vector<float>> userFactors;
    std::vector<std::vector<float>> itemFactors;
    std::vector<float> userBiases;
    std::vector<float> itemBiases;
    float globalMean = 0.0f;

    // NCF components 
    std::vector<std::vector<float>> userEmbeddings;
    std::vector<std::vector<float>> itemEmbeddings;

    // User-Item interactions for IBCF
    std::vector<std::unordered_set<int>> userInteractions; // Stores items each user has interacted with

    // Hyperparameters
    const int SVD_FACTORS = 5;
    const int NCF_EMBEDDING_DIM = 3;
    const float SVD_LR = 0.0178f;
    const float NCF_LR = 0.04f;
    const float REG = 0.052f;
    const int EPOCHS = 44;
    const float WEIGHTSVD = 0.7f;
    const float WEIGHTNCF = 0.3f; // 1.f - WEIGHTSVD
    const float WEIGHTIBCF = 0; // IBCF not used, therefore 0 weight 
    const int IBCF_TOP_K = 10; // Number of top similar items to consider

    std::mt19937 rng{ 10486 };

    void initializeModels(size_t numUsers, size_t numItems) {
        std::normal_distribution<float> dist(-0.01f, 0.01f);

        // Initialize SVD components
        userFactors.resize(numUsers, std::vector<float>(SVD_FACTORS));
        itemFactors.resize(numItems, std::vector<float>(SVD_FACTORS));
        userBiases.resize(numUsers, 0.0f);
        itemBiases.resize(numItems, 0.0f);

        // Initialize NCF components
        userEmbeddings.resize(numUsers, std::vector<float>(NCF_EMBEDDING_DIM));
        itemEmbeddings.resize(numItems, std::vector<float>(NCF_EMBEDDING_DIM));

        // Initialize User-Item interactions
        userInteractions.resize(numUsers);

        // Initialize all vectors with small random values
        for (auto vec : { &userFactors, &itemFactors, &userEmbeddings, &itemEmbeddings }) {
            for (auto& row : *vec) {
                for (float& val : row) {
                    val = dist(rng);
                }
            }
        }
    }

    float predictSVD(int userId, int itemId) const {
        float pred = globalMean + userBiases[userId] + itemBiases[itemId];
        for (int f = 0; f < SVD_FACTORS; f++) {
            pred += userFactors[userId][f] * itemFactors[itemId][f];
        }
        return pred;
    }

    float predictNCF(int userId, int itemId) const {
        return std::inner_product(
            userEmbeddings[userId].begin(),
            userEmbeddings[userId].end(),
            itemEmbeddings[itemId].begin(),
            0.0f
        );
    }

    // Cosine Similarity method
    float cosineSimilarity(const std::vector<float>& vec1, const std::vector<float>& vec2) const {
        if (vec1.empty() || vec2.empty() || vec1.size() != vec2.size()) {
            return 0.0f;
        }

        float dotProduct = 0.0f;
        float normVec1 = 0.0f;
        float normVec2 = 0.0f;

        for (size_t i = 0; i < vec1.size(); ++i) {
            dotProduct += vec1[i] * vec2[i];
            normVec1 += vec1[i] * vec1[i];
            normVec2 += vec2[i] * vec2[i];
        }

        if (normVec1 == 0.0f || normVec2 == 0.0f) {
            return 0.0f;
        }

        return dotProduct / (std::sqrt(normVec1) * std::sqrt(normVec2));
    }

    // Item-Based Collaborative Filtering (IBCF) method
    float IBCF(int userId, int itemId) const {
        if (userId >= static_cast<int>(userInteractions.size())) {
            return 0.0f;
        }

        const auto& interactedItems = userInteractions[userId];
        if (interactedItems.empty()) {
            return 0.0f;
        }

        // Vector to store similarities and corresponding ratings
        std::vector<std::pair<float, float>> similarities;

        for (const auto& interactedItemId : interactedItems) {
            // Compute similarity between target item and interacted item
            float similarity = cosineSimilarity(itemFactors[interactedItemId], itemFactors[itemId]);
            if (similarity > 0.0f) { // Consider only positive similarities
                similarities.emplace_back(similarity, 1.0f); // Assuming binary interactions; adjust if ratings are available
            }
        }

        if (similarities.empty()) {
            return 0.0f;
        }

        // Sort similarities in descending order and take top K
        std::sort(similarities.begin(), similarities.end(),
            [](const std::pair<float, float>& a, const std::pair<float, float>& b) {
                return a.first > b.first;
            });

        size_t topK = std::min(static_cast<size_t>(IBCF_TOP_K), similarities.size());

        float weightedSum = 0.0f;
        float similaritySum = 0.0f;

        for (size_t i = 0; i < topK; ++i) {
            weightedSum += similarities[i].first * similarities[i].second; // Multiply similarity by interaction (e.g., rating)
            similaritySum += similarities[i].first;
        }

        if (similaritySum == 0.0f) {
            return 0.0f;
        }

        return weightedSum / similaritySum;
    }

public:
    void train(const std::vector<std::tuple<int, int, float>>& data) {

        size_t numUsers = 0;
        size_t numItems = 0;
        float sumRatings = 0.0f;

        // Determine the number of users and items
        for (size_t idx = 0; idx < data.size(); ++idx) {
            int u = std::get<0>(data[idx]);
            int i = std::get<1>(data[idx]);
            float r = std::get<2>(data[idx]);
            numUsers = std::max(numUsers, static_cast<size_t>(u + 1));
            numItems = std::max(numItems, static_cast<size_t>(i + 1));
            sumRatings += r;
        }

        globalMean = sumRatings / data.size();
        initializeModels(numUsers, numItems);

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            float rmse = 0.0f;

            for (size_t idx = 0; idx < data.size(); ++idx) {
                int userId = std::get<0>(data[idx]);
                int itemId = std::get<1>(data[idx]);
                float rating = std::get<2>(data[idx]);

                // Compute hybrid prediction
                float svdPred = predictSVD(userId, itemId);
                float ncfPred = predictNCF(userId, itemId);

                //IBCF Prediction
                //float ibcfPred = IBCF(userId, itemId);

                float hybridPred = WEIGHTSVD * svdPred + WEIGHTNCF * ncfPred /* + WEIGHTIBCF * ibcfPred */;
                float err = rating - hybridPred;
                rmse += err * err;

                // Update SVD components
                float uBias = userBiases[userId];
                float iBias = itemBiases[itemId];
                userBiases[userId] += SVD_LR * (err * WEIGHTSVD - REG * uBias);
                itemBiases[itemId] += SVD_LR * (err * WEIGHTSVD - REG * iBias);

                for (int f = 0; f < SVD_FACTORS; f++) {
                    float uFactor = userFactors[userId][f];
                    float iFactor = itemFactors[itemId][f];

                    userFactors[userId][f] += SVD_LR * (err * WEIGHTSVD * iFactor - REG * uFactor);
                    itemFactors[itemId][f] += SVD_LR * (err * WEIGHTSVD * uFactor - REG * iFactor);
                }

                // Update NCF components
                for (int d = 0; d < NCF_EMBEDDING_DIM; d++) {
                    float uEmbed = userEmbeddings[userId][d];
                    float iEmbed = itemEmbeddings[itemId][d];

                    userEmbeddings[userId][d] += NCF_LR * (err * WEIGHTNCF * iEmbed - REG * uEmbed);
                    itemEmbeddings[itemId][d] += NCF_LR * (err * WEIGHTNCF * uEmbed - REG * iEmbed);
                }


                // User-Item interactions for IBCF 
                /*
                userInteractions[userId].insert(itemId);
                */
            }
            /* Debugging for RMSE calculation in each epoch */

            if (epoch % 5 == 0 || epoch == EPOCHS - 1) {
                rmse = std::sqrt(rmse / data.size());
                std::cerr << "Epoch " << epoch + 1 << " RMSE: " << rmse << "\n";
            }
            
        }
    }

    float predictRating(int userId, int itemId) const {
        float svdPred = predictSVD(userId, itemId);
        float ncfPred = predictNCF(userId, itemId);
        // IBCF not used for better performance
        // float ibcfPred = IBCF(userId, itemId);
        float hybridPred = (WEIGHTSVD * svdPred) + (WEIGHTNCF * ncfPred) /* +(WEIGHTIBCF * ibcfPred) */;
        return std::max(0.0f, std::min(hybridPred, 5.0f));
    }

    void processTrainingData(const std::string& trainingFile, const std::string& testFile) {
        std::ifstream trainingInput(trainingFile);
        if (!trainingInput.is_open()) {
            std::cerr << "Error: Unable to open " << trainingFile << "\n";
            return;
        }

        std::vector<std::tuple<int, int, float>> data;
        std::string line;

        // Skip header
        std::getline(trainingInput, line);

        while (std::getline(trainingInput, line)) {
            std::stringstream ss(line);
            int userId, itemId;
            float rating;
            ss >> userId >> itemId >> rating;

            data.emplace_back(std::make_tuple(userId, itemId, rating));
        }

        trainingInput.close();
        train(data);

        std::ifstream testInput(testFile);
        if (!testInput.is_open()) {
            std::cerr << "Error: Unable to open " << testFile << "\n";
            return;
        }

        std::ofstream output("predicted_ratings.csv");
        if (!output.is_open()) {
            std::cerr << "Error: Unable to open predicted_ratings.csv\n";
            return;
        }
        output << "userId,itemId,predictedRating\n";

        while (std::getline(testInput, line)) {
            std::stringstream ss(line);
            int userId, itemId;
            ss >> userId >> itemId;

            if (userId >= userFactors.size() || itemId >= itemFactors.size()) {
                std::cerr << "Invalid userId or itemId in test data\n";
                continue;
            }

            float predictedRating = predictRating(userId, itemId);
            output << userId << "," << itemId << "," 
                << std::fixed << std::setprecision(1) << predictedRating << "\n";
        }

        testInput.close();
        output.close();
    }

};

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    HybridRecommender recommender;
    recommender.processTrainingData("training_data.csv", "test_data.csv");

    return 0;
}
