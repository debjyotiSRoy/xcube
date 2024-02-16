#!/bin/bash

# Path to the directory containing the shell scripts
script_directory="launches"
infer=false

# Check if the script directory exists
if [ ! -d "$script_directory" ]; then
	echo "Script directory '$script_directory' does not exist."
	exit 1
fi

# Check if the script_list file argument is provided
if [ "$#" -lt 1 ]; then
	echo "Usage: $0 --script_list_file <file> [--infer] [--output_file]"
	exit 1
fi

while [[ $# -gt 0 ]]; do 
	case "$1" in
		-f|--script_list_file)
			script_list_file="$2"
			shift 2
			;;
		-i|--infer)
			infer=true
			shift
			;;
		-o|--output_file)
            if [ "$infer" = true ]; then
                output_file="$2"
                shift 2
            else
                echo "Error: The --output_file parameter can only be used with --infer."
                exit 1
            fi
            ;;
		*)
			echo "Unknown option: $1"
			exit 1
			;;
	esac
done

# Check if --infer is present but --output_file is missing
if [ "$infer" = true ] && [ -z "$output_file" ]; then
    echo "Error: When using --infer, you must also provide an --output_file."
    exit 1
fi

# Check if the script list file exists
if [ ! -f "$script_list_file" ]; then
	echo "Script list file '$script_list_file' does not exist."
	exit 1
fi

if [ "$infer" = true ]; then
	# Create the output directory if it doesn't exist
	output_dir=$(dirname "$output_file")
	mkdir -p "$output_dir"
    echo "Script, Loss, Precision_at_1, Precision_at_2, Precision_at_3, Precision_at_5, Recall_at_25, Recall_at_30, Recall_at_35, Recall_at_40, Recall_at_50" > "$output_file"
	# fi
fi

extract_metric() {
    grep -oP "$1 = \K[0-9.]+" <<< "$2"
}

# Run and process each shell script
run_and_process_script() {
    script_name="$1"
	script_parameters="$2"  # Additional parameters to pass to the script
    # output="$("$script_name")"
	# Run the script, capture its output, and display it on the console using tee
    output="$("$script_name" $script_parameters | tee /dev/tty)"
    
    loss=$(extract_metric "test_loss" "$output")
    precision_at_5=$(extract_metric "precision_at_5" "$output")
    precision_at_8=$(extract_metric "precision_at_8" "$output")
    precision_at_15=$(extract_metric "precision_at_15" "$output")
    f1_score_macro=$(extract_metric "f1_score_macro" "$output")
    f1_score_micro=$(extract_metric "f1_score_micro" "$output")
    
    # Append the metrics to the CSV file
    echo "$script_name, $loss, $precision_at_5, $precision_at_8, $precision_at_15, $f1_score_macro, $f1_score_micro" >> "$output_file"
}

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
			if [ "$infer" = true ]; then
				run_and_process_script "./$script_directory/$script_name" "$parameters"
			else
				"./$script_directory/$script_name" $parameters
			fi
		else
			echo "Script '$script_name' not found or is not a regular file."
		fi
done < "$script_list_file"