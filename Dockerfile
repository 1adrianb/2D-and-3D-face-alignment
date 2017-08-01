FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04 

WORKDIR  /opt
RUN git clone https://github.com/torch/distro.git ~/torch --recursive \
    && cd torch

RUN bash install-deps
RUN ./install.sh




