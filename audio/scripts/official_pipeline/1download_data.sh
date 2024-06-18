#!/bin/bash

# Navigate to the audio directory
cd ${MY_HOME} || exit 1

# Ensure the data directory exists and enter it
mkdir -p data
cd data || exit 1

# Download and extract dataset function
function download_and_extract() {
    local dataset_url="https://zenodo.org/record/1237703/files"
    local file_name="$1"

    echo "Checking and downloading $file_name..."

    # Only download if the file does not exist
    if [ ! -f "$file_name" ]; then
        wget "$dataset_url/$file_name" || { echo "Failed to download $file_name"; exit 1; }
    fi

    # Unzip and remove the archive if the download was successful
    unzip "$file_name" -d "ansim/" && rm "$file_name"
}

# Ensure the ansim directory exists
mkdir -p ansim

# License file check
if [ ! -f LICENSE ]; then
    wget "https://zenodo.org/record/1237703/files/LICENSE" || { echo "Failed to download LICENSE"; exit 1; }
fi

# Download and extract all required files
for split in {1..3}; do
    for ov in {1..3}; do
        download_and_extract "ov${ov}_split${split}.zip"
    done
done

echo "All files downloaded and extracted."

# Return to the original directory
popd || exit 1
