#!/bin/bash

# Default number of parallel processes
num_processes=5
version="default_version"

# Parse command-line arguments
while getopts "p:v:" opt; do
  case $opt in
    p)
      num_processes=$OPTARG
      ;;
    v)
      version=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# Define the split keys and batch indices
split_keys=("held_out" "random_1" "random_2" "random_3" "random_4" "random_5")
batch_indices=$(seq 0 49)

# Total number of processes to run
total_processes=$((${#split_keys[@]} * 50))
completed_processes=0
completed_processes_file=$(mktemp)
echo 0 > "$completed_processes_file"
start_time=$(date +%s)

echo "Total tasks to complete: $total_processes"

run_command() {
  split_key=$1
  batch_idx=$2
  python features/save_ml_input.py --dataset mimic --version "$version" --split_key "$split_key" --batch_idx "$batch_idx" > /dev/null 2>&1
  completed_processes=$(($(cat "$completed_processes_file") + 1))
  echo "$completed_processes" > "$completed_processes_file"
  
  current_time=$(date +%s)
  elapsed_time=$((current_time - start_time))
  progress=$(awk "BEGIN {printf \"%.1f\", ($completed_processes * 100 / $total_processes)}")
  estimated_total_time=$((elapsed_time * total_processes / completed_processes))
  remaining_time=$((estimated_total_time - elapsed_time))
  
  # Convert remaining time to hours, minutes, and seconds
  hours=$((remaining_time / 3600))
  minutes=$(( (remaining_time % 3600) / 60 ))
  seconds=$((remaining_time % 60))
  
  echo -ne "Progress: $progress% complete. Estimated time remaining: ${hours}h ${minutes}m ${seconds}s\r"
}

export -f run_command
export total_processes
export start_time
export completed_processes_file
export version

# Generate combinations of split_key and batch_idx and run in parallel
for split_key in "${split_keys[@]}"; do
  for batch_idx in $batch_indices; do
    echo "$split_key $batch_idx"
  done
done | xargs -n 2 -P "$num_processes" -I {} bash -c 'run_command $@' _ {}

echo -e "\nAll processes completed."
rm "$completed_processes_file"
