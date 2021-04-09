# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install any needed packages specified in requirements.txt and Stanford CoreNLP.
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt autoremove --purge \
    && python -m spacy download en_core_web_sm \
    && wget http://nlp.stanford.edu/software/stanford-corenlp-4.2.0.zip \
    && unzip stanford-corenlp-4.2.0.zip \
    && cd stanford-corenlp-4.2.0 \
    && for file in `find . -name "*.jar"`; do export CLASSPATH="$CLASSPATH:`realpath $file`"; done \
    && git clone --recursive https://github.com/MILVLG/bottom-up-attention.pytorch \
    && cd detectron2 \
    && pip install -e . \
    && cd .. \
    && $ git clone https://github.com/NVIDIA/apex.git \
    && cd apex \
    && python setup.py install --cuda_ext --cpp_ext \
    && cd .. \
    && python setup.py build develop \
    && cd .. 

# Run when the container launches
CMD CUDA_VISIBLE_DEVICES=0 python test.py