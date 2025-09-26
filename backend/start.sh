#!/bin/bash
echo "=== KrishiMitra Startup Script ==="
echo "Python version: $(python --version)"
echo "Current directory: $(pwd)"
echo "Files in directory:"
ls -la
echo "Environment variables:"
echo "PORT: $PORT"
echo "RAILWAY_ENVIRONMENT: $RAILWAY_ENVIRONMENT"
echo "Starting application..."
exec gunicorn --bind 0.0.0.0:$PORT --timeout 120 --workers 1 --log-level info app:app
