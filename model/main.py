import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Download NLTK resources if not already downloaded
nltk.download('wordnet')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('/content/news.csv')
print("Check for duplicates:", df.duplicated().sum())
print("\nCheck for missing values:")
print(df.isnull().sum())
df['date'] = pd.to_datetime(df['date'], errors='coerce')
print(df.info())

# Visualize distribution of sentiment
df['sentiment'].value_counts().plot(kind='bar')
plt.title('Distribution of Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Preprocess text in the "news" column
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

df['clean_news'] = df['news'].apply(preprocess_text)

# Encode sentiment labels (POSITIVE=1, NEGATIVE=0)
le = LabelEncoder()
df['sentiment_encoded'] = le.fit_transform(df['sentiment'])

# Split data into training and testing sets
X = df['clean_news'].values
y = df['sentiment_encoded'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Tokenization and padding parameters
max_features = 10000  # Top 10,000 words
maxlen = 100          # Maximum sequence length
tokenizer = Tokenizer(num_words=max_features, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen, padding='post', truncating='post')

# Build improved model architecture using Bidirectional LSTM
embedding_dim = 128
model = Sequential([
    Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=maxlen),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(32)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(X_train_pad, y_train,
                    epochs=10,
                    batch_size=64,
                    validation_data=(X_test_pad, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test_pad, y_test)
print("Improved Test accuracy: {:.2f}%".format(accuracy * 100))

# Save the model and tokenizer
model.save("sentiment_model.h5")
with open("tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
