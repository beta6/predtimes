#!/bin/bash
# predict.sh
PROJECT_ID=$1
NUM_PREDICTIONS=$2
# Activate virtual environment if necessary
# source /path/to/your/venv/bin/activate
python3 -m celery -A predtimes.celery call main.tasks.generate_predictions_task --args="[$PROJECT_ID, $NUM_PREDICTIONS]"
