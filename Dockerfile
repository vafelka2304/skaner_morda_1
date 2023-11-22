FROM python:3.11.5 as builder

RUN apt-get update && apt-get install -y python3 python3-pip

CMD ["python", "-u","./deep_fake,py"]