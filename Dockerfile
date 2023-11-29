# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Upgrade pipenv
RUN pip install --upgrade pipenv

# Install system dependencies
RUN apt-get update \
    && apt-get install -y libsndfile1 libsndfile1-dev

# Install Python dependencies
RUN pipenv install

# Install additional dependencies (if needed)
RUN pip install pysoundfile

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV LIBROSA_HINT=libsndfile
#RUN ["pipenv","run","python", "./vgg16_model.py"]

# Run app.py when the container launches
CMD ["pipenv", "run", "python", "app.py"]

