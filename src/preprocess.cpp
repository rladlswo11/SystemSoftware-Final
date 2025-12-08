/// @file preprocess.cpp
/// @brief Implementation of the preprocess executable.
#include "preprocess.hpp"
#include "common.hpp"
#include "math_layer.hpp"

#include <fstream>
#include <iostream>
#include <string>

namespace preprocess
{
    int run(const std::string &csv_path)
    {
        std::ifstream in(csv_path);
        if (!in)
        {
            std::cerr << "preprocess: failed to open CSV file: "
                      << csv_path << std::endl;
            return 1;
        }

        std::string line;
        while (std::getline(in, line))
        {
            if (line.empty())
            {
                continue;
            }

            common::Sample s{};
            if (!common::parse_csv_line(line, s))
            {
                std::cerr << "preprocess: failed to parse line: " << line << std::endl;
                continue;
            }

            // Normalize features using math layer.
            math::normalize_sample(s);

            // Output whitespace-separated line to stdout.
            std::cout << common::sample_to_line(s) << '\n';
        }

        return 0;
    }

} // namespace preprocess

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: preprocess <csv_path>\n";
        return 1;
    }
    return preprocess::run(argv[1]);
}
