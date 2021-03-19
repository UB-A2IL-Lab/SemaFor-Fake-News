sudo docker build -t ub/image-text:v1 .

# # export docker image
# sudo docker save -o docker_image.tar ub/image-text:v1.0

# # publish docker image
# sudo docker login
# sudo docker tag docker_sample YOUR_DOCKER_ID/docker_sample
# sudo docker push YOUR_DOCKER_ID/docker_sample