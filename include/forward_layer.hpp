/// @file forward_layer.hpp
/// @brief Interface for the forward_layer executable.
#pragma once

namespace forward_layer
{
    /**
     * @brief Run the forward feature processing stage.
     *
     * Reads normalized samples from stdin, performs simple feature
     * augmentation, and writes them to stdout in the same format.
     *
     * @return 0 on success, non-zero on error.
     */
    int run();
} // namespace forward_layer
