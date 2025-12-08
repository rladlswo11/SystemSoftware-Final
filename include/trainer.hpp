/// @file trainer.hpp
/// @brief Interface for the trainer executable (orchestrator).
#pragma once

#include <string>

namespace trainer
{
    /**
     * @brief Run the trainer (orchestrator).
     *
     * Creates pipes, forks the four stages:
     *   preprocess -> forward_layer -> backward_layer -> logger
     *
     * and manages their lifetime.
     *
     * @param csv_path Path to the input CSV dataset.
     * @return 0 on success, non-zero on error.
     */
    int run(const std::string &csv_path);
} // namespace trainer
