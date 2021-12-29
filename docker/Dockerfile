FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

RUN python3 -m pip install click==7.0 
RUN python3 -m pip install pytorch-lightning==1.0.8
RUN python3 -m pip install torch-summary==1.4.3
RUN python3 -m pip install comet-ml==3.2.6
RUN python3 -m pip install segmentation_models_pytorch==0.1.0
RUN conda install -y opencv

RUN mkdir -p /home/root/datasets

RUN apt-get update
RUN apt-get install -y zsh

RUN python3 -m pip install pandas==1.1.4
ENV MKL_THREADING_LAYER=GNU

EXPOSE 6006

ENTRYPOINT ["/bin/zsh"]
