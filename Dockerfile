FROM tensorflow/tensorflow:1.15.5
RUN pip install keras==2.3.1 scikit_learn==0.22.1 dataclasses networkx==2.2 pandas==0.25.3 xgboost==1.0.1 anytree==2.8.0 
RUN pip install h2o==3.28.1.2 tpot==0.11.1 statsmodels==0.11.1 matplotlib==3.0.3 Pillow==7.0.0 imageio==2.8.0
RUN apt-get _update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install opencv-python

