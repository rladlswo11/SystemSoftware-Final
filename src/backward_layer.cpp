/// @file backward_layer.cpp
/// @brief Implementation of the backward_layer executable.
#include "backward_layer.hpp"
#include "common.hpp"
#include "math_layer.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace
{
    /// @brief Operating mode for the backward layer.
    enum class Mode
    {
        Train, ///< Training mode (update parameters).
        Test   ///< Test mode (no updates, just evaluate).
    };

    /**
     * @brief Obtain mode from BACKWARD_MODE environment variable.
     *
     * Accepted values:
     *   - "test" or "TEST" -> Mode::Test
     *   - anything else or unset -> Mode::Train
     *
     * @return Parsed mode.
     */
    Mode get_mode()
    {
        const char *env = std::getenv("BACKWARD_MODE");
        if (!env || !*env)
        {
            return Mode::Train;
        }

        const std::string s(env);
        if (s == "test" || s == "TEST")
        {
            return Mode::Test;
        }
        return Mode::Train;
    }

    /**
     * @brief Get model parameter file path from MODEL_FILE or default.
     *
     * If MODEL_FILE is set and non-empty, it is used as the path.
     * Otherwise, "logs/model_params.txt" is used.
     *
     * @return Path to the parameter file.
     */
    std::string get_model_path()
    {
        const char *env = std::getenv("MODEL_FILE");
        if (env && *env)
        {
            return std::string(env);
        }
        return "logs/model_params.txt";
    }

    /**
     * @brief Streaming loop implementing the backward stage.
     *
     * In train mode:
     *   - forward + backward + parameter update
     *   - output "id loss y_hat"
     *
     * In test mode:
     *   - forward only, compute loss = 0.5 * (y_hat - y)^2
     *   - output "id loss y_hat"
     *
     * @param mode Selected operating mode.
     * @return 0 on success, non-zero on error.
     */
    int run_stream(Mode mode)
    {
        std::ios::sync_with_stdio(false);

        std::string line;
        std::size_t count = 0;

        while (std::getline(std::cin, line))
        {
            if (line.empty())
            {
                continue;
            }

            common::Sample s{};
            if (!common::parse_sample_line(line, s))
            {
                std::cerr << "backward_layer: failed to parse line: "
                          << line << '\n';
                continue;
            }

            float y_hat = 0.0f;
            math::compute_forward(s, y_hat);

            float loss = 0.0f;
            float grad_norm = 0.0f;

            if (mode == Mode::Train)
            {
                math::compute_backward_and_update(s, y_hat, loss, grad_norm);
            }
            else
            {
                const float diff = y_hat - s.y;
                loss = 0.5f * diff * diff;
                grad_norm = 0.0f;
            }

            ++count;
            std::cout << s.id << ' ' << loss << ' ' << y_hat << '\n';
        }

        return 0;
    }

} // namespace

namespace backward_layer
{
    int run()
    {
        const Mode mode = get_mode();
        const std::string model_path = get_model_path();

        if (mode == Mode::Test)
        {
            if (!math::load_parameters(model_path))
            {
                std::cerr << "backward_layer: no model file at "
                          << model_path
                          << ", using initial parameters\n";
            }
            else
            {
                std::cerr << "backward_layer: loaded parameters from "
                          << model_path << '\n';
            }
        }

        const int rc = run_stream(mode);

        if (mode == Mode::Train)
        {
            if (!math::save_parameters(model_path))
            {
                std::cerr << "backward_layer: failed to save parameters to "
                          << model_path << '\n';
            }
            else
            {
                std::cerr << "backward_layer: saved parameters to "
                          << model_path << '\n';
            }
        }

        return rc;
    }

} // namespace backward_layer

int main()
{
    return backward_layer::run();
}
