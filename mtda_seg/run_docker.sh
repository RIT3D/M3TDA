# docker build -t donghe/ubuntu_mmdet_${1} .

docker run -it \
        --gpus all \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v /ssdfei/Datasets/MTDA2:/data \
        -v /home/rcv_js/ws/MTDA:/workspace/ \
        --privileged \
        --network=host \
        --ipc=host \
        donghe/ubuntu_mmdet_${1}