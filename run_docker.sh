# Author: Panfei, Donghe
# Feature: Development environment 
# Usage:
#        bash run_docker.sh m3tda

docker build -t donghe/ubuntu_mmdet_${1} .

# #Fei fan
# docker run -it \
#         --gpus all \
#         -v /tmp/.X11-unix:/tmp/.X11-unix \
#         -v /ssdfei/Datasets/MTDA2:/data \
#         -v /home/rcv_js/ws/MTDA:/workspace/ \
#         --privileged \
#         --network=host \
#         --ipc=host \
#         donghe/ubuntu_mmdet_${1}

#server | desktop_4090ti
docker run -it \
        --gpus all \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v /data:/data \
        -v /home/dhe/M3TDA:/workspace \
        --privileged \
        --network=host \
        --ipc=host \
        donghe/ubuntu_mmdet_${1}
