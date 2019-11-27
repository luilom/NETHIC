# Use an official Python runtime as a parent image
FROM python:3.6.3

# Set the working directory to /workdir
WORKDIR /workdir

# Path to read Neural Networks and dictionaries
ENV pathToReadDataset /categorizer-vol

# Copy the current directory contents into the container at /workdir
ADD . /workdir

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# RUN mkdir to create dir with volume contents
RUN mkdir /workdir/categorizer-vol

# Make port 81 available to the world outside this container
EXPOSE 81

# Run python script when the container launches
CMD ["python3", "StartClassifier.py", "81", "categorizer-vol", "count", "simple"]
