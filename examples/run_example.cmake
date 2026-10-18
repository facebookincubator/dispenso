# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Runs one example as a CTest smoke test. Invoked as
#   cmake -DEXAMPLE=<path> [-DEXPECT_DONE=ON] -P run_example.cmake
#
# An example passes when it exits with status 0, every "<got> (expected: <want>)"
# line it prints has <got> equal to <want>, and, with EXPECT_DONE, it reaches its
# closing "completed successfully" line. The examples are written for reading,
# not as tests, so these checks only use what they already print.

if(NOT EXAMPLE)
  message(FATAL_ERROR "run_example.cmake: EXAMPLE is not set")
endif()

execute_process(
  COMMAND "${EXAMPLE}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE output)

# Echo the output so a failure's log shows what the example printed.
message("${output}")

set(failures "")
if(NOT result STREQUAL "0")
  list(APPEND failures "exited with '${result}'")
endif()

string(REGEX MATCHALL "[^\n]*\\(expected:? [^)\n]*\\)" checks "${output}")
foreach(line IN LISTS checks)
  if(line MATCHES "([-+.0-9eE]+) \\(expected:? ([-+.0-9eE]+)\\)$")
    if(NOT CMAKE_MATCH_1 STREQUAL CMAKE_MATCH_2)
      list(APPEND failures "mismatch: '${line}'")
    endif()
  endif()
endforeach()

if(EXPECT_DONE AND NOT output MATCHES "completed successfully")
  list(APPEND failures "never printed its 'completed successfully' line")
endif()

if(failures)
  list(JOIN failures "\n  " text)
  message(FATAL_ERROR "${EXAMPLE} failed:\n  ${text}")
endif()
