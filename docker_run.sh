# # load docker image from local machine
# sudo docker load -i docker_image.tar
# # pull docker image from docker hub
# docker pull yjzhux/image-text:1.0-neuralnews-test

# run docker
# make sure you've doloaded all the required data and map the path to the folder
# of the docker (/app/data)
sudo docker run -v /mnt/data/NeuralNews/data/:/app/data \
                -it --rm --gpus all \
                ub/image-text:v1.0 \
                /bin/bash
# After launching the docker, run the foloowing commands to test demo data.
# export CLASSPATH=$CLASSPATH:./stanford-corenlp-4.2.0/stanford-corenlp-4.2.0.jar 
# CUDA_VISIBLE_DEVICES=0 python test.py