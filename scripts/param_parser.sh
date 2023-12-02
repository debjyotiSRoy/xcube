# Function to parse named parameters
parse_params() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -f|--fname)
                fname="$2"
                shift 2
                ;;
            -p|--plant)
                plant=true
                shift
                ;;
            -d|--diff_inattn)
                diff_inattn="$2"
                shift 2
                ;;
            -l|--l2r_sgdr_lr0)
                l2r_sgdr_lr0="$2"
                shift 2
                ;; 
            -i|--infer)
                if [[ "$2" == "0" || "$2" == "1" ]]; then
                    infer="$2"  # Assign the value of the next argument to the 'infer' variable
                    shift 2      # Shift 2 arguments, as we've consumed both the option and its value
                else
                    echo "Invalid value for --infer: $2. Should be 0 or 1."
                    exit 1
                fi
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
done
}