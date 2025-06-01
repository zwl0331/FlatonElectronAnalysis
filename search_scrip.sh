#!/usr/bin/env bash

# Directory you want to scan:
search_dir="/pnfs/mu2e/persistent/users/mu2epro/workflow/default/outstage/77020096/00"

# Where to save the list:
output_file="root_paths.txt"

# Start fresh
> "$output_file"

# Recursively find all .root files under $search_dir, suppress permission warnings,
# and append their absolute paths to the output file
find "$search_dir" -type f -name '*.root' 2>/dev/null >> "$output_file"

# Summarize
echo "Found $(wc -l < "$output_file") .root files under:"
echo "  $search_dir"
echo "Paths written to $(realpath "$output_file")"