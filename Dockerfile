FROM pytorch/pytorch:latest
RUN apt-get update && apt-get -y install libglib2.0-dev libsm6 libxext6 libxrender1
RUN pip install Cython && pip install matplotlib scikit-learn opencv-python dill
