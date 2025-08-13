#!/bin/bash

#################################
# Author:       Matej Ciglenečki (modified)
# Description:  Recursively moves files and directories to new subdirectories.
#               Each subdirectory will contain a max specified number of items.
#################################

# Directory containing your flat collection
if [ "$#" -ne 2 ]; then
    echo "Error: provide arguments ./subdir_mover.sh <target_dir> <num_items_per_subdir>"
fi
target_dir=$1

# Max number of items (files + directories) per subdirectory
num_items_per_subdir=$2

# Check for existing numbered directories from 0 to num_items_per_subdir
for i in $(seq 0 $num_items_per_subdir); do
    if [[ -d "$target_dir/$i" ]]; then
        echo "Error: Directory '$target_dir/$i' already exists. Please remove or rename it before running this script."
        exit 1
    fi
done

# Counters for the number of items processed and current subdirectory index
counter=0
subdir_index=0

# Create the first subdirectory
mkdir -p "$target_dir/$subdir_index"

# Process every item (file or directory) in the target directory (excluding the created subdirectories)
find "$target_dir" -maxdepth 1 ! -name "$(basename "$target_dir")" ! -name "[0-9]*" | sort | while read -r item; do

    # Skip the subdirectories we create for moving
    if [[ "$item" == "$target_dir/$subdir_index" ]]; then
        continue
    fi

    # Check if item still exists (not moved yet)
    if [[ -e "$item" ]]; then

        # If we've reached the limit for the current subdirectory, move to the next one
        if (( counter > 0 && counter % num_items_per_subdir == 0 )); then
            subdir_index=$((subdir_index + 1))
            mkdir -p "$target_dir/$subdir_index"
        fi

        # Move the item (file or directory)
        mv "$item" "$target_dir/$subdir_index/"

        # Increment counter
        counter=$((counter + 1))
    fi
done
