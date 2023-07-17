FROM python:3.9.6
WORKDIR /api
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY setup.py .
COPY ml/ ./ml
EXPOSE 8000
CMD ["uvicorn", "setup:app", "--host", "0.0.0.0"]
