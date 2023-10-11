#!/bin/bash

# Setup
# Clone barretenberg repository (https://github.com/aztecprotocol/barretenberg/)
# Copy this script to barretenberg/cpp

# Usage
# cd barretenberg/cpp
# ./bootstrap.sh
# ./list_tests_barretenberg.sh

# Location of binaries
BIN_DIR="./build/bin/"

# Iterate over all "_tests" binaries
for test_bin in "${BIN_DIR}"*_tests; do
    # Check if file exists and is executable
    if [[ -f "$test_bin" && -x "$test_bin" ]]; then
        echo "Listing tests from $test_bin"
        "$test_bin" --gtest_list_tests
    fi
done
