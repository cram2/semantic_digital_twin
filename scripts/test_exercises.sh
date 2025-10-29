#!/bin/bash
source /opt/ros/jazzy/setup.bash
# Determine the directory of this script and change to the examples directory relative to it
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXERCISES_DIR="$(cd "$SCRIPT_DIR/../self_assessment/exercises" && pwd)"
cd "$EXERCISES_DIR"
rm -rf test_tmp
mkdir test_tmp
jupytext --to notebook *.md
mv *.ipynb test_tmp
cd test_tmp
treon --thread 1 -v