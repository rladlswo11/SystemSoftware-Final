/// @file math_layer.hpp
/// @brief Public API for preprocessing and backpropagation math.
#pragma once

#include <cstddef>
#include <string>
#include "common.hpp"

namespace math
{
    /**
     * @brief Normalize a single sample in-place.
     *
     * Uses internally defined per-feature mean and standard deviation.
     *
     * @param s Sample to normalize.
     */
    void normalize_sample(common::Sample &s);

    /**
     * @brief Augment features (e.g., simple nonlinear transformation).
     *
     * For this demo, we apply a light transformation to features
     * (e.g., square the last feature). This simulates feature engineering.
     *
     * @param s Sample to modify.
     */
    void augment_features(common::Sample &s);

    /**
     * @brief Compute the forward pass for a single sample.
     *
     * The model is a small fully connected network:
     *   input (dimension INPUT_DIM) -> hidden layer (ReLU) -> scalar output.
     *
     * @param s      Input sample (features assumed normalized/augmented).
     * @param y_hat  Output prediction.
     */
    void compute_forward(const common::Sample &s, float &y_hat);

    /**
     * @brief Compute backward pass and update model parameters.
     *
     * Uses mean squared error loss:
     *   L = 0.5 * (y_hat - y)^2
     *
     * Performs one step of SGD on the internal parameters.
     *
     * @param s          Input sample (features + label).
     * @param y_hat      Prediction from forward pass.
     * @param loss_out   Output loss for this sample.
     * @param grad_norm  Output L2 norm of the gradient.
     */
    void compute_backward_and_update(const common::Sample &s,
                                     float y_hat,
                                     float &loss_out,
                                     float &grad_norm);

    /**
     * @brief Save current model parameters to a text file.
     *
     * The file format is a simple human-readable text format and is
     * only intended to be used by this program.
     *
     * @param path Path to the file (will be overwritten).
     * @return true on success, false on failure.
     */
    bool save_parameters(const std::string &path);

    /**
     * @brief Load model parameters from a text file.
     *
     * If the file does not exist or has incompatible dimensions,
     * this function returns false and leaves the current parameters
     * unchanged.
     *
     * @param path Path to the file.
     * @return true on success, false on failure.
     */
    bool load_parameters(const std::string &path);
} // namespace math
