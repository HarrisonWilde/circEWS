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

# Total number of processes to run
total_processes=50
completed_processes_file=$(mktemp)
echo 0 > "$completed_processes_file"
start_time=$(date +%s)

echo "Total tasks to complete: $total_processes"

run_command() {
  idx=$1
  python circulatory_status/make_grid.py --idx "$idx" --version "$version" --reduced --mimic > /dev/null 2>&1
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

# Generate indices and run in parallel
seq 0 49 | xargs -n 1 -P "$num_processes" -I {} bash -c 'run_command $@' _ {}

echo -e "\nAll processes completed."
rm "$completed_processes_file"
