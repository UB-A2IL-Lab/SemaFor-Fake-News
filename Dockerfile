# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt autoremove --purge

# Run when the container launches
CMD CUDA_VISIBLE_DEVICES=0 python train.py