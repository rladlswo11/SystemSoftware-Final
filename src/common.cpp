/// @file common.cpp
/// @brief Implementation of shared utility functions.
#include "common.hpp"

#include <sstream>

namespace common
{
    bool parse_csv_line(const std::string &line, Sample &out)
    {
        std::stringstream ss(line);
        char comma;

        if (!(ss >> out.id)) return false;
        if (!(ss >> comma) || comma != ',') return false;
        for (std::size_t i = 0; i < INPUT_DIM; ++i)
        {
            if (!(ss >> out.x[i])) return false;
            if (i < INPUT_DIM - 1)
            {
                if (!(ss >> comma) || comma != ',') return false;
            }
        }
        if (!(ss >> comma) || comma != ',') return false;
        if (!(ss >> out.y)) return false;

        return true;
    }

    std::string sample_to_line(const Sample &s)
    {
        std::ostringstream oss;
        oss << s.id;
        for (std::size_t i = 0; i < INPUT_DIM; ++i)
        {
            oss << ' ' << s.x[i];
        }
        oss << ' ' << s.y;
        return oss.str();
    }

    bool parse_sample_line(const std::string &line, Sample &out)
    {
        std::stringstream ss(line);
        if (!(ss >> out.id)) return false;
        for (std::size_t i = 0; i < INPUT_DIM; ++i)
        {
            if (!(ss >> out.x[i])) return false;
        }
        if (!(ss >> out.y)) return false;
        return true;
    }

} // namespace common
