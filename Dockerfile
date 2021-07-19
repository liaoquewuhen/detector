FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel

ENV LISTEN_PORT 5000
EXPOSE 5000

COPY ./mnist1d.py /app/
COPY ./mnist1d_detection_notzip.pkl /app/
COPY ./detectFlask.py /app/
WORKDIR /app

RUN pip install flask
RUN pip install opencv-python

CMD ["python", "detectFlask.py"]