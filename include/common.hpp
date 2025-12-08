/// @file common.hpp
/// @brief Common constants, data structures, and utilities shared across components.
#pragma once

#include <cstddef>
#include <string>

namespace common
{
    /// @brief Dimensionality of the input feature vector.
    constexpr std::size_t INPUT_DIM = 4;

    /// @brief Simple sample structure: features and scalar label.
    struct Sample
    {
        float x[INPUT_DIM]; ///< Normalized (and possibly augmented) input features.
        float y;            ///< Target label.
        int   id;           ///< Sample identifier (line number in CSV, 1-based).
    };

    /**
     * @brief Parse a CSV line into a Sample structure.
     *
     * Expected CSV format:
     *   id, f0, f1, f2, f3, label
     *
     * All fields are required.
     *
     * @param line     Input CSV line.
     * @param out      Output sample (filled on success).
     * @return true on successful parse, false otherwise.
     */
    bool parse_csv_line(const std::string &line, Sample &out);

    /**
     * @brief Convert a Sample to a whitespace-separated string.
     *
     * Format:
     *   id f0 f1 f2 f3 y
     *
     * Used for piping between processes.
     *
     * @param s Sample to convert.
     * @return String representation.
     */
    std::string sample_to_line(const Sample &s);

    /**
     * @brief Parse a whitespace-separated line into a Sample.
     *
     * Expected format:
     *   id f0 f1 f2 f3 y
     *
     * @param line Input line.
     * @param out  Output sample.
     * @return true on success, false otherwise.
     */
    bool parse_sample_line(const std::string &line, Sample &out);

} // namespace common


