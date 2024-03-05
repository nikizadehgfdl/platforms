## Docker general
sudo docker container ls
sudo docker stop a1138e62a17d 007975c9140b
sudo docker rm a1138e62a17d 007975c9140b
sudo docker ps --filter "status=exited"
sudo docker container prune
sudo docker volume ls --filter "dangling=true"
sudo docker volume prune
sudo docker volume rm docker_nextcloud

## How to add external storage:
### Creat/populate/chown the directory to be exposed
sudo mkdir /media/nextcloud_docker
sudo mkdir /media/nextcloud_docker/readings
sudo cp -r /home/niki/nikiHomeShare/readings/papers /media/nextcloud_docker/readings/
sudo chown -R www-data:www-data /media/nextcloud_docker

## How to bring it up
sudo docker-compose build --pull
sudo docker-compose  up -d


