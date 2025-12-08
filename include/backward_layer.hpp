/// @file backward_layer.hpp
/// @brief Interface for the backward_layer executable.
#pragma once

namespace backward_layer
{
    /**
     * @brief Run the backward pass stage.
     *
     * Reads samples with (possibly augmented) features from stdin.
     * Behavior is controlled by the BACKWARD_MODE environment variable:
     *
     *   - BACKWARD_MODE=train (default):
     *       Perform forward + backward passes, update parameters,
     *       and write:
     *         id loss y_hat
     *       for each sample to stdout.
     *       At the end of the stream, the trained parameters are saved
     *       to MODEL_FILE (or logs/model_params.txt by default).
     *
     *   - BACKWARD_MODE=test:
     *       Load parameters from MODEL_FILE if present, perform forward
     *       passes only (no parameter updates), compute loss, and write:
     *         id loss y_hat
     *       for each sample to stdout.
     *
     * @return 0 on success, non-zero on error.
     */
    int run();
} // namespace backward_layer
