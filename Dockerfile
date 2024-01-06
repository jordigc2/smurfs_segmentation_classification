FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel as smurfs_image
ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils

RUN  apt-get update -y \
&&   apt-get install -y \
     software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa \
&&  apt-get update \
&&  apt-get install -y \
    python3-pip \
    git \
    python3.6-tk \
    python3-opencv
    
RUN pip3 install -U pip
RUN pip3 install -U PyYAML --ignore-installed
RUN pip3 install opencv-python

COPY ./requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt