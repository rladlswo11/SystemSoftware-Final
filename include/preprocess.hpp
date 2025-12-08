/// @file preprocess.hpp
/// @brief Interface for the preprocess executable.
#pragma once

#include <string>

namespace preprocess
{
    /**
     * @brief Run the preprocessing stage.
     *
     * Reads a CSV dataset file, parses and normalizes samples, and writes
     * whitespace-separated samples to stdout.
     *
     * @param csv_path Path to CSV file.
     * @return 0 on success, non-zero on error.
     */
    int run(const std::string &csv_path);
} // namespace preprocess
