# Use an official Python runtime as a parent image
FROM python:3.6.3

# Set the working directory to /workdir
WORKDIR /workdir

# Copy the current directory contents into the container at /workdir
ADD . /workdir

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# RUN to create a shared folder on Volume
RUN mkdir /workdir/categorizer-vol


# Run python script when the container launches
CMD ["python3", "Training.py", "/workdir/categorizer-vol/datasets_training", "count", "categorizer-vol" ]