/// @file math_layer.cpp
/// @brief Implementation of preprocessing and backpropagation math.

/// In a real ML system, these matrix-heavy operations would usually run on
/// a GPU or an optimized multi-threaded library (BLAS, cuBLAS, etc.).
/// Here we intentionally keep them as simple single-threaded CPU code so
/// that the assignment can focus on process-level parallelism (fork, pipes,
/// signals) rather than numeric library internals.

#include "math_layer.hpp"

#include <array>
#include <cmath>
#include <fstream>

namespace
{
    using common::INPUT_DIM;
    using common::Sample;

    /// @brief Hidden layer size for the small neural network.
    constexpr std::size_t HIDDEN_DIM = 8;

    /// @brief Learning rate for SGD.
    constexpr float LEARNING_RATE = 0.001f;

    /// @brief Per-feature mean (for normalization).
    ///
    /// The synthetic data is generated with features ~ N(0, 1),
    /// so the natural normalization is centered at 0 with unit variance.
    constexpr std::array<float, INPUT_DIM> FEATURE_MEAN = {
        0.0f, 0.0f, 0.0f, 0.0f};

    /// @brief Per-feature standard deviation (for normalization).
    constexpr std::array<float, INPUT_DIM> FEATURE_STD = {
        1.0f, 1.0f, 1.0f, 1.0f};

    /// @brief Input-to-hidden weights W1[j][k] (j: hidden, k: input).
    std::array<std::array<float, INPUT_DIM>, HIDDEN_DIM> g_W1 = {{
        {{ 0.10f,  0.00f,  0.00f,  0.00f }},
        {{ 0.00f,  0.10f,  0.00f,  0.00f }},
        {{ 0.00f,  0.00f,  0.10f,  0.00f }},
        {{ 0.00f,  0.00f,  0.00f,  0.10f }},
        {{-0.10f,  0.00f,  0.00f,  0.00f }},
        {{ 0.00f, -0.10f,  0.00f,  0.00f }},
        {{ 0.00f,  0.00f, -0.10f,  0.00f }},
        {{ 0.00f,  0.00f,  0.00f, -0.10f }},
    }};

    /// @brief Hidden biases b1[j].
    std::array<float, HIDDEN_DIM> g_b1 = {
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f};

    /// @brief Hidden-to-output weights w2[j].
    std::array<float, HIDDEN_DIM> g_w2 = {
        0.05f,  0.05f,  0.05f,  0.05f,
       -0.05f, -0.05f, -0.05f, -0.05f};

    /// @brief Output bias.
    float g_b2 = 0.0f;

    /**
     * @brief Apply per-feature normalization.
     *
     * x[i] <- (x[i] - mean[i]) / std[i].
     */
    void normalize_features(float *x)
    {
        for (std::size_t i = 0; i < INPUT_DIM; ++i)
        {
            x[i] = (x[i] - FEATURE_MEAN[i]) / FEATURE_STD[i];
        }
    }

    /**
     * @brief Simple feature augmentation.
     *
     * For this demo, square the last feature to simulate a nonlinearity.
     */
    void augment(float *x)
    {
        if (INPUT_DIM > 0)
        {
            x[INPUT_DIM - 1] = x[INPUT_DIM - 1] * x[INPUT_DIM - 1];
        }
    }

    /**
     * @brief ReLU activation function.
     *
     * @param z Pre-activation value.
     * @return max(0, z).
     */
    inline float relu(float z)
    {
        return (z > 0.0f) ? z : 0.0f;
    }

    /**
     * @brief Forward pass through hidden and output layers.
     *
     * @param s    Input sample (features in s.x).
     * @param z1   Output pre-activations of hidden layer (size HIDDEN_DIM).
     * @param a1   Output activations of hidden layer (size HIDDEN_DIM).
     * @param yhat Output scalar prediction.
     */
    void forward_all(const Sample &s,
                     std::array<float, HIDDEN_DIM> &z1,
                     std::array<float, HIDDEN_DIM> &a1,
                     float &yhat)
    {
        // Hidden layer
        for (std::size_t j = 0; j < HIDDEN_DIM; ++j)
        {
            float z = g_b1[j];
            for (std::size_t k = 0; k < INPUT_DIM; ++k)
            {
                z += g_W1[j][k] * s.x[k];
            }
            z1[j] = z;
            a1[j] = relu(z);
        }

        // Output layer (linear)
        float z2 = g_b2;
        for (std::size_t j = 0; j < HIDDEN_DIM; ++j)
        {
            z2 += g_w2[j] * a1[j];
        }
        yhat = z2;
    }

    /**
     * @brief Compute loss and gradients, and update parameters.
     *
     * Uses MSE loss and SGD on the 2-layer network:
     *   L = 0.5 * (y_hat - y)^2.
     *
     * @param s            Input sample (features + label).
     * @param y_hat        Prediction from forward pass.
     * @param z1           Hidden pre-activations (size HIDDEN_DIM).
     * @param a1           Hidden activations (size HIDDEN_DIM).
     * @param loss_out     Output loss.
     * @param grad_norm_out Output L2 norm of the gradient.
     */
    void backward_internal(const Sample &s,
                           float y_hat,
                           const std::array<float, HIDDEN_DIM> &z1,
                           const std::array<float, HIDDEN_DIM> &a1,
                           float &loss_out,
                           float &grad_norm_out)
    {
        const float diff = y_hat - s.y;

        // Loss: 0.5 * (y_hat - y)^2
        loss_out = 0.5f * diff * diff;

        // Output layer gradients
        float dL_dz2 = diff;

        std::array<float, HIDDEN_DIM> dL_dw2{};
        for (std::size_t j = 0; j < HIDDEN_DIM; ++j)
        {
            dL_dw2[j] = dL_dz2 * a1[j];
        }
        float dL_db2 = dL_dz2;

        // Backprop into hidden layer
        std::array<float, HIDDEN_DIM> dL_dz1{};
        for (std::size_t j = 0; j < HIDDEN_DIM; ++j)
        {
            float dL_da1 = dL_dz2 * g_w2[j];
            dL_dz1[j] = (z1[j] > 0.0f) ? dL_da1 : 0.0f;
        }

        // Input-to-hidden gradients
        std::array<std::array<float, INPUT_DIM>, HIDDEN_DIM> dL_dW1{};
        std::array<float, HIDDEN_DIM> dL_db1{};

        for (std::size_t j = 0; j < HIDDEN_DIM; ++j)
        {
            dL_db1[j] = dL_dz1[j];
            for (std::size_t k = 0; k < INPUT_DIM; ++k)
            {
                dL_dW1[j][k] = dL_dz1[j] * s.x[k];
            }
        }

        // Gradient norm
        double norm_sq = 0.0;

        for (std::size_t j = 0; j < HIDDEN_DIM; ++j)
        {
            for (std::size_t k = 0; k < INPUT_DIM; ++k)
            {
                const float g = dL_dW1[j][k];
                norm_sq += static_cast<double>(g) * g;
            }
            const float gb1 = dL_db1[j];
            const float gw2 = dL_dw2[j];
            norm_sq += static_cast<double>(gb1) * gb1;
            norm_sq += static_cast<double>(gw2) * gw2;
        }
        norm_sq += static_cast<double>(dL_db2) * dL_db2;

        grad_norm_out = static_cast<float>(std::sqrt(norm_sq));

        // Parameter update (SGD)
        for (std::size_t j = 0; j < HIDDEN_DIM; ++j)
        {
            for (std::size_t k = 0; k < INPUT_DIM; ++k)
            {
                g_W1[j][k] -= LEARNING_RATE * dL_dW1[j][k];
            }
            g_b1[j] -= LEARNING_RATE * dL_db1[j];
            g_w2[j] -= LEARNING_RATE * dL_dw2[j];
        }
        g_b2 -= LEARNING_RATE * dL_db2;
    }

} // namespace

namespace math
{
    void normalize_sample(common::Sample &s)
    {
        normalize_features(s.x);
    }

    void augment_features(common::Sample &s)
    {
        augment(s.x);
    }

    void compute_forward(const common::Sample &s, float &y_hat)
    {
        std::array<float, HIDDEN_DIM> z1{};
        std::array<float, HIDDEN_DIM> a1{};
        forward_all(s, z1, a1, y_hat);
    }

    void compute_backward_and_update(const common::Sample &s,
                                     float y_hat,
                                     float &loss_out,
                                     float &grad_norm)
    {
        std::array<float, HIDDEN_DIM> z1{};
        std::array<float, HIDDEN_DIM> a1{};
        float y_hat_check = 0.0f;
        forward_all(s, z1, a1, y_hat_check);
        (void)y_hat; // not used; could be checked against y_hat_check if desired.

        backward_internal(s, y_hat_check, z1, a1, loss_out, grad_norm);
    }

    bool save_parameters(const std::string &path)
    {
        std::ofstream ofs(path);
        if (!ofs)
        {
            return false;
        }

        ofs << HIDDEN_DIM << ' ' << INPUT_DIM << '\n';

        // W1
        for (std::size_t j = 0; j < HIDDEN_DIM; ++j)
        {
            for (std::size_t k = 0; k < INPUT_DIM; ++k)
            {
                ofs << g_W1[j][k] << ' ';
            }
            ofs << '\n';
        }

        // b1
        for (std::size_t j = 0; j < HIDDEN_DIM; ++j)
        {
            ofs << g_b1[j] << ' ';
        }
        ofs << '\n';

        // w2
        for (std::size_t j = 0; j < HIDDEN_DIM; ++j)
        {
            ofs << g_w2[j] << ' ';
        }
        ofs << '\n';

        // b2
        ofs << g_b2 << '\n';

        return static_cast<bool>(ofs);
    }

    bool load_parameters(const std::string &path)
    {
        std::ifstream ifs(path);
        if (!ifs)
        {
            return false;
        }

        std::size_t hidden_dim = 0;
        std::size_t input_dim = 0;
        ifs >> hidden_dim >> input_dim;
        if (!ifs || hidden_dim != HIDDEN_DIM || input_dim != INPUT_DIM)
        {
            return false;
        }

        // W1
        for (std::size_t j = 0; j < HIDDEN_DIM; ++j)
        {
            for (std::size_t k = 0; k < INPUT_DIM; ++k)
            {
                ifs >> g_W1[j][k];
            }
        }

        // b1
        for (std::size_t j = 0; j < HIDDEN_DIM; ++j)
        {
            ifs >> g_b1[j];
        }

        // w2
        for (std::size_t j = 0; j < HIDDEN_DIM; ++j)
        {
            ifs >> g_w2[j];
        }

        // b2
        ifs >> g_b2;

        return static_cast<bool>(ifs);
    }

} // namespace math
