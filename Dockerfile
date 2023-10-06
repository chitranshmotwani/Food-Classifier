# Use a base image with Python and necessary dependencies
FROM python:3.11-slim-buster

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install required dependencies
RUN apt-get update && apt-get install -y gcc python3-dev
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your application runs on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py"]
