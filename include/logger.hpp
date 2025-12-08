/// @file logger.hpp
/// @brief Interface for the logger executable.
#pragma once

namespace logger
{
    /**
     * @brief Run the logging stage.
     *
     * Reads lines of the form:
     *   id loss y_hat
     *
     * from stdin, maintains summary statistics, and reacts to signals:
     *   - SIGUSR1: print intermediate statistics to stderr.
     *   - SIGTERM: request graceful termination (flush and exit).
     *
     * @return 0 on success, non-zero on error.
     */
    int run();
} // namespace logger
