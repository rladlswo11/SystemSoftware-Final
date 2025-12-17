/// @file logger.cpp
/// @brief Implementation of the logger executable.
#include "logger.hpp"

#include <atomic>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace
{
    /// @brief Set to true when a snapshot dump is requested (SIGUSR1).
    std::atomic<bool> g_dump_requested{false};

    /// @brief Set to true when a graceful termination is requested (SIGTERM).
    std::atomic<bool> g_terminate_requested{false};

    /**
     * @brief Signal handler for SIGUSR1.
     *
     * Requests a snapshot dump on the next iteration of the main loop.
     */
    void handle_sigusr1(int)
    {
        g_dump_requested.store(true);
    }

    /**
     * @brief Signal handler for SIGTERM.
     *
     * Requests termination after the current iteration.
     */
    void handle_sigterm(int)
    {
        g_terminate_requested.store(true);
    }
} // namespace

namespace logger
{
    int run()
    {
        std::ios::sync_with_stdio(false);

        // Install signal handlers.
        std::signal(SIGUSR1, handle_sigusr1);
        std::signal(SIGTERM, handle_sigterm);

        std::size_t count = 0;
        double total_loss = 0.0;
        double total_yhat = 0.0;

        std::string line;
        while (std::getline(std::cin, line))
        {
            if (line.empty())
            {
                if (g_terminate_requested.load())
                {
                    break;
                }
                continue;
            }

            int id = 0;
            float loss = 0.0f;
            float y_hat = 0.0f;

            {
                std::stringstream ss(line);
                if (!(ss >> id >> loss >> y_hat))
                {
                    std::cerr << "logger: failed to parse line: "
                              << line << std::endl;
                    continue;
                }
            }

            ++count;
            total_loss += static_cast<double>(loss);
            total_yhat += static_cast<double>(y_hat);

            // Per-sample line for downstream logging / progress.
            std::cout << "SAMPLE " << id
                      << " LOSS " << loss
                      << " YHAT " << y_hat << '\n';

            if (g_dump_requested.load())
            {
                g_dump_requested.store(false);
                if (count > 0)
                {
                    const double avg_loss = total_loss / static_cast<double>(count);
                    const double avg_yhat = total_yhat / static_cast<double>(count);
                    std::cerr << "[LOGGER SNAPSHOT] samples=" << count
                              << " avg_loss=" << avg_loss
                              << " avg_yhat=" << avg_yhat << std::endl;
                }
            }

            if (g_terminate_requested.load())
            {
                break;
            }
        }

        // Final summary.
        if (count > 0)
        {
            const double avg_loss = total_loss / static_cast<double>(count);
            const double avg_yhat = total_yhat / static_cast<double>(count);

            // Machine-readable summary on stdout for the shell script:
            // SUMMARY <samples> <avg_loss> <avg_yhat>
            std::cout << "SUMMARY "
                      << count << ' '
                      << std::setprecision(6) << avg_loss << ' '
                      << std::setprecision(6) << avg_yhat << '\n';
        }
        else
        {
            // No samples processed; report to stderr for debugging.
            std::cerr << "[LOGGER FINAL] no samples processed" << std::endl;
        }

        return 0;
    }

} // namespace logger

int main()
{
    return logger::run();
}
