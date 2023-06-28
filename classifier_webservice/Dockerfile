FROM python:3.9-slim
COPY ./app.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./iris_trained_model.pkl /deploy/
WORKDIR /deploy/
RUN pip3 install -r requirements.txt
EXPOSE 2000
ENTRYPOINT [ "python3", "app.py" ]
