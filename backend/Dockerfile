FROM python:3.11-slim

WORKDIR /code

# Copy requirements and install
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy backend source code and model checkpoint
COPY . /code

# Hugging Face Spaces exposes port 7860 by default
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
