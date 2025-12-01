#!/bin/bash

# Start the Django development server in the background
python predtimes/manage.py runserver 0.0.0.0:8000 &

cd predtimes
# Start the Celery worker in the foreground
exec celery -A predtimes worker -l info
