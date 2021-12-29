docker rm sbin
docker build --tag=sbin ./docker/
docker run -it --ipc=host --gpus all -p 6006:6006 -v $PWD:/workspace -v $PWD:/datasets --name sbin sbin
