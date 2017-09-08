FROM kaixhin/cuda-torch

# Install depenencies and python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python-numpy \
    python-matplotlib \
    libmatio2 \
    libgoogle-glog-dev \
    libboost-all-dev \
    python-dev \
    python-tk

RUN pip install dlib

# Install lua packages
RUN luarocks install xlua &&\
    luarocks install matio

# Build thpp
WORKDIR /opt
RUN git clone https://github.com/facebook/thpp
WORKDIR /opt/thpp
RUN git fetch origin pull/33/head:NEWBRANCH && git checkout NEWBRANCH
WORKDIR /opt/thpp/thpp
RUN THPP_NOFB=1 ./build.sh

# Build fb.python
WORKDIR /opt
RUN git clone https://github.com/facebook/fblualib
WORKDIR /opt/fblualib/fblualib/python
RUN luarocks make rockspec/*

# Clone our repo
WORKDIR /workspace
RUN chmod -R a+w /workspace
RUN git clone https://github.com/1adrianb/2D-and-3D-face-alignment