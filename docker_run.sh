# # load docker image
# sudo docker load -i docker_image.tar

# run docker
sudo docker run -v /mnt/data/NeuralNews/data/:/app/data \
                -it --rm --gpus all \
                ub/image-text:v1