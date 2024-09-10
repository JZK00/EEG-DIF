#!/bin/bash

# Check if a command argument was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 '<command_to_run>'"
    exit 1
fi

COMMAND="$1"

# Function to check GPU usage
check_gpu_usage() {
    # Using nvidia-smi to get GPU utilization (%), checking if any GPU is above 50%
    if nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | awk '{if($1 > 50) exit 1}'; then
        return 1 # GPU usage is high, indicating training might be running
    else
        return 0 # GPU usage is low
    fi
}

# Function to execute the given command
run_command() {
    echo "Executing command: $COMMAND"
    eval $COMMAND
    echo "Command finished."
}

# Main monitoring loop
while true; do
    if ! check_gpu_usage; then
        echo "GPU usage is low. Executing the command immediately."
        run_command
        # After executing the command once, continue monitoring GPU usage.
        exit 0
    else
        echo "High GPU usage detected. Waiting for it to decrease..."
        # Wait until GPU usage is low
        while ! check_gpu_usage; do
            sleep 10  # Check every 10 seconds
        done
    fi

    echo "Do you want to execute the command \"$COMMAND\"? (yes/no, you have 5 minutes to answer)"
    read -t 300 input  # Wait for user input for up to 5 minutes

    if [[ $input == "yes" ]]; then
        run_command
        exit 0
    elif [[ $input == "no" ]]; then
        echo "Sleeping for 20 minutes..."
        sleep 1200  # Sleep for 20 minutes
    else
        echo "No input received, or input not recognized. Not executing the command."
        # No sleep here, immediately return to monitoring.
    fi
done
