# kneenet-docker

## Requirements

[docker](https://www.docker.com/) or [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

## Installation

    git clone https://github.com/stanfordnmbl/kneenet-docker
    
## Usage

Enter `kneenet-docker` directory. Copy knee x-rays to `input` directory.
Run

    sudo bash run.sh
    
Results will be available in `output`

## Building docker

    sudo docker build . -t stanford-nmbl/kneenet
