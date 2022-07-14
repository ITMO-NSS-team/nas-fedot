FROM nvcr.io/nvidia/rapidsai/rapidsai:21.06-cuda11.2-base-ubuntu20.04
RUN mkdir -p /home/hdd/
WORKDIR /home/hdd/
RUN mkdir docker
COPY . ./docker
RUN apt install sudo git
