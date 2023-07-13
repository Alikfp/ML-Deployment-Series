FROM python:3.9.6
WORKDIR /api
COPY requirements.txt /api/requirements.txt
RUN pip3 install -r api/requirements.txt
COPY ml api/
EXPOSE 8000
CMD ["uvicorn", "setup:app", "--host", "0.0.0.0"]
