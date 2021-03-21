# Trained models are included. (./run/models/)
sudo docker build -t ub/image-text:v1.0 .

# # export docker image
# sudo docker save -o docker_image.tar ub/image-text:v1.0

# # publish docker image
# sudo docker login
# sudo docker tag ub/image-text:v1.0 YOUR_DOCKER_ID/image-text:1.0-neuralnews-test
# sudo docker push YOUR_DOCKER_ID/image-text:1.0-neuralnews-test