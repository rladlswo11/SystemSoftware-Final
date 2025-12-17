/// @file forward_layer.cpp
/// @brief Implementation of the forward_layer executable.
#include "forward_layer.hpp"
#include "common.hpp"
#include "math_layer.hpp"

#include <iostream>
#include <string>

namespace forward_layer
{
    int run()
    {
        std::ios::sync_with_stdio(false);

        std::string line;
        while (std::getline(std::cin, line))
        {
            if (line.empty())
            {
                continue;
            }

            common::Sample s{};
            if (!common::parse_sample_line(line, s))
            {
                std::cerr << "forward_layer: failed to parse line: " << line << std::endl;
                continue;
            }

            // Feature augmentation (e.g., simple nonlinearity).
            math::augment_features(s);

            // Pass augmented sample forward in the pipeline.
            std::cout << common::sample_to_line(s) << '\n';
        }

        return 0;
    }

} // namespace forward_layer

int main()
{
    return forward_layer::run();
}
