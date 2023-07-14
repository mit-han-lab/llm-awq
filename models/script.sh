#!/bin/bash

# List all directories
directories=$(find . -type d)

# Loop through each directory
for dir in $directories; do
  # Change to the directory
  cd "$dir" || continue

  # Check if there are any changes to be committed, excluding Git LFS tracked files
  if [[ $(git status --porcelain --ignore=submodule,untracked -uno) ]]; then
    # Add all changes
    git add .

    # Commit changes
    git commit -m "Updated README"

    # Push changes
    git push
  fi

  # Change back to the previous directory
  cd ..
done

