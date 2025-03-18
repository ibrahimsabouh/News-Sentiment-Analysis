FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app/ ./app/
COPY model/sentiment_model.h5 ./model/
COPY model/tokenizer.pickle ./model/

# Set environment variables
ENV MODEL_PATH=./model/sentiment_model.h5
ENV TOKENIZER_PATH=./model/tokenizer.pickle

# Expose the port the app runs on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]