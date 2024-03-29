FROM python:3.9-slim-buster
WORKDIR /app
COPY app.py /app/app.py
COPY model/model.joblib /app/models/model.joblib
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
CMD ["python","app.py"]