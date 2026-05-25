FROM python:3.11.8

WORKDIR /backend/app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]