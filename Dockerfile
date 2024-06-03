FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
RUN apt update -y
RUN apt upgrade -y
RUN pip install torch torchvision numpy wandb tqdm scikit-learn matplotlib
