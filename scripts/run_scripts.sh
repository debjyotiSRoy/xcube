#!/bin/bash

# Path to the directory containing the shell scripts
script_directory="launches"

# Check if the script directory exists
if [ ! -d "$script_directory" ]; then
  echo "Script directory '$script_directory' does not exist."
  exit 1
fi

# Check if the script_list file argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <script_list_file>"
    exit 1
fi

# Read the list of script names from a file
script_list_file="$1"

# Check if the script list file exists
if [ ! -f "$script_list_file" ]; then
  echo "Script list file '$script_list_file' does not exist."
  exit 1
fi

# Loop through each script name in the list and run it
# while IFS= read -r script_name; do
#   # Check if the script file exists
#   if [ -f "$script_directory/$script_name" ]; then
#     # Make the script file executable (if it's not already)
#     chmod +x "$script_directory/$script_name"

#     # Run the script
#     echo "Running script: $script_name"
#     "$script_directory/$script_name"
#   else
#     echo "Script '$script_name' not found or is not a regular file."
#   fi
# done < "$script_list_file"

# Loop through each line in the list
while IFS= read -r line; do
    # Extract the script name and parameters
    script_name=$(echo "$line" | awk '{print $1}')
    parameters=$(echo "$line" | awk '{for (i=2; i<=NF; i++) print $i}')

    # Check if the script file exists
    if [ -f "$script_directory/$script_name" ]; then
        # Make the script file executable (if it's not already)
        chmod +x "$script_directory/$script_name"

        # Run the script with extracted parameters
        "./$script_directory/$script_name" $parameters
    else
        echo "Script '$script_name' not found or is not a regular file."
    fi
done < "$script_list_file"
