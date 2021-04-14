# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install any needed packages specified in requirements.txt and Stanford CoreNLP.
RUN apt-get update && apt-get -y upgrade \
    && apt-get install -y unzip ffmpeg libsm6 libxext6 openjdk-8-jdk\
    && apt-get autoremove --purge \
    && pip install --upgrade pip \
    && pip install -r requirements.txt \
    && python -m spacy download en_core_web_sm \
    && curl -O -L http://nlp.stanford.edu/software/stanford-corenlp-latest.zip \
    && unzip stanford-corenlp-latest.zip \
    && export CLASSPATH=$CLASSPATH:./stanford-corenlp-4.2.0/stanford-corenlp-4.2.0.jar \
    && rm stanford-corenlp-latest.zip
# Build and install faster r-cnn to rextract image feature.
RUN git clone --recursive https://github.com/MILVLG/bottom-up-attention.pytorch \
    && cd bottom-up-attention.pytorch/detectron2 \
    && pip install -e . \
    && cd .. \
    && git clone https://github.com/NVIDIA/apex.git \
    && curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/libcublas-dev_10.1.0.105-1_amd64.deb \
    && dpkg -i libcublas-dev_10.1.0.105-1_amd64.deb && rm libcublas-dev_10.1.0.105-1_amd64.deb\
    && cd apex \
    && python setup.py install --cuda_ext --cpp_ext \
    && cd .. \
    && python setup.py build develop \
    && cd .. 

# Run when the container launches
# CMD CUDA_VISIBLE_DEVICES=0 python test.py