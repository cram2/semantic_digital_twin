#!/bin/bash
# Determine the directory of this script and change to the exercises directory relative to it
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/../self_assessment/exercises" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/configs/nb_remove_solutions.json"

echo "Using config file: $CONFIG_FILE"

cd "$EXAMPLES_DIR" || exit 1

# Clean and recreate tmp directory
rm -rf converted_exercises
mkdir converted_exercises

# Convert all MyST markdown files to notebooks
jupytext --to notebook *.md

# Move notebooks into tmp for processing
mv *.ipynb converted_exercises
cd converted_exercises || exit 1

# Run nbconvert on each notebook with the shared config
for nb in *.ipynb; do
  echo "Converting $nb ..."
  jupyter nbconvert --config "$CONFIG_FILE" --to notebook --inplace "$nb"
done
