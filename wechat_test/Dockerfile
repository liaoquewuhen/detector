FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel



COPY / /app

WORKDIR /app

RUN pip install flask
RUN pip install opencv-python

CMD ["python", "detectFlask.py"]