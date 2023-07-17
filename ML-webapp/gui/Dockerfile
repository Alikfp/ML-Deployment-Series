FROM python:3.9.6
WORKDIR /gui
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY ./gui.py .
EXPOSE 8501
CMD ["streamlit", "run", "gui.py"]