# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for some ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Expose port 7860 which is required by Hugging Face Spaces
EXPOSE 7860

# Run the FastAPI application using Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
