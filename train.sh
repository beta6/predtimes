#!/bin/bash
# train.sh

# Validate that both PROJECT_ID and ARCHITECTURE are provided
if [ "$#" -ne 2 ]; then
    echo "Uso: $0 <PROJECT_ID> <ARCHITECTURE>"
    echo "Ejemplo: $0 123 \"resnet50\""
    exit 1
fi

PROJECT_ID=$1
ARCHITECTURE=$2

# Activate virtual environment if necessary
# source /path/to/your/venv/bin/activate
python3 -m celery -A predtimes.celery call main.tasks.train_model_task --args="[$PROJECT_ID, \"$ARCHITECTURE\"]"
