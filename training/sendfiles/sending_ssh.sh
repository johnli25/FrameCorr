#!/bin/bash

# Configuration
SOURCE_FOLDER="."
TARGET_USER="deepakn3"
TARGET_HOST="172.22.154.247"
TARGET_FOLDER="/home/deepakn3/Progressive-Neural-Compression/training/recv_files"

# Check for required arguments
if [ -z "$SOURCE_FOLDER" ] || [ -z "$TARGET_USER" ] || [ -z "$TARGET_HOST" ] || [ -z "$TARGET_FOLDER" ]; then
    echo "Usage: $0 <source_folder> <target_user> <target_host> <target_folder>"
    exit 1
fi

# Loop through files in the source folder
for file in "$SOURCE_FOLDER"/*; do
    if [ -f "$file" ]; then  # Make sure it's a regular file
        echo "Transferring $file"
        scp "$file" "$TARGET_USER@$TARGET_HOST:$TARGET_FOLDER"
    fi
done
