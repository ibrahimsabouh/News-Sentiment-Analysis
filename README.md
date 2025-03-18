# News Sentiment Analysis

A machine learning application that analyzes the sentiment of news text using a BiLSTM neural network model.

## Features

- Text preprocessing with NLTK (removing stopwords, lemmatization)
- Sentiment classification (Positive/Negative)
- Interactive web interface built with Streamlit
- Containerized with Docker for easy deployment

## Project Structure

```
News-Sentiment-Analysis/
├── app/                   # Web application
│   ├── static/            # CSS and static files
│   ├── templates/         # HTML templates (if needed)
│   ├── app.py             # Streamlit application
│   └── requirements.txt   # App-specific dependencies
├── model/                 # Model training and artifacts
│   ├── main.py            # Training script
│   ├── sentiment_model.h5 # Trained model
│   └── tokenizer.pickle   # Fitted tokenizer
├── Dockerfile             # Docker configuration
├── .dockerignore          # Files to exclude from Docker
├── .gitignore             # Files to exclude from Git
├── README.md              # Project documentation
└── requirements.txt       # Project dependencies
```

## Installation

### Local Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/News-Sentiment-Analysis.git
   cd News-Sentiment-Analysis
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   streamlit run app/app.py
   ```

### Docker Setup

1. Build the Docker image:
   ```
   docker build -t news-sentiment-analysis .
   ```

2. Run the container:
   ```
   docker run -p 8501:8501 news-sentiment-analysis
   ```

3. Access the application at `http://localhost:8501`

## Usage

1. Enter news text in the provided text area
2. Click the "Analyze" button
3. View sentiment prediction results (Positive/Negative)

## Model Training

The model was trained on a news dataset with sentiment labels. To retrain the model:

```
python model/main.py
```

This will:
- Load and preprocess the dataset
- Build a BiLSTM neural network
- Train the model
- Save the model and tokenizer

## Docker Image

The application is available as a Docker image on DockerHub:

```
docker pull yourusername/news-sentiment-analysis
docker run -p 8501:8501 yourusername/news-sentiment-analysis
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

[MIT License](LICENSE)

## Author

Ibrahim Sabouh