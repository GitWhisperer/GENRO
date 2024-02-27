import numpy as np
import librosa
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, utils
from keras.utils import to_categorical
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import sounddevice as sd
from pydub import AudioSegment

# Spotify credentials
SPOTIFY_CID = "__"  
SPOTIFY_SECRET = "__"

# List of genres
genres = ['pop', 'rock', 'blues', 'jazz', 'country', 'heavy_metal', 'electronic', 'folk']

# Function to load audio from URL using pydub
def load_audio_from_url(url):
    audio = AudioSegment.from_url(url)
    samples = np.array(audio.get_array_of_samples())
    return samples, audio.frame_rate

# Load data
X, y = load_data()

# Check if the dataset is not empty before performing train-test split
if len(X) == 0:
    print("Dataset is empty. Unable to perform train-test split.")
else:
    # Perform train-test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

# Spotify helper functions
def load_data():
    X = []
    y = []
    # Spotify authentication
    client_credentials_manager = SpotifyClientCredentials(client_id=SPOTIFY_CID, client_secret=SPOTIFY_SECRET)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    for genre in genres:
        results = sp.search(q=f'genre:{genre}')
        for track in results['tracks']['items']:
            track_id = track['id']
            # Get track details including the full audio file URL
            track_details = sp.track(track_id)
            audio_url = track_details['preview_url']
            # Check if the track has a preview URL
            if audio_url:
                try:
                    samples, sample_rate = load_audio_from_url(audio_url)
                    if len(samples) == 0:
                        print(f"Error loading audio at URL: {audio_url}: Empty audio snippet.")
                        continue
                    X.append(samples)
                    y.append(genre)
                except Exception as e:
                    print(f"Error loading audio at URL: {audio_url}: {str(e)}")
                    continue

    return np.array(X), np.array(y)



# Split data into training, validation, and test sets
def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, y_train, X_val, y_val, X_test, y_test

# Preprocess data
def preprocess_data(X, y):
    data = []
    labels = []

    for i in range(len(X)):
        try:
            # Download the entire audio file using librosa
            snippet, _ = librosa.load(librosa.example(X[i]), sr=None)
            if len(snippet) == 0:
                print(f"Error loading audio at index {i}: Empty audio snippet.")
                continue
            data.append(snippet)
            labels.append(y[i])
        except Exception as e:
            print(f"Error loading audio at index {i}: {str(e)}")
            continue

    # Convert labels to categorical
    labels = to_categorical(labels, num_classes=len(set(y)))

    # Convert data and labels to NumPy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# Build model
def build_model(input_shape, n_classes):
    model = keras.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(n_classes, activation='softmax'))
    return model

# Train model
def train_model(model, X_train, y_train, X_val, y_val):
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              batch_size=32,
              epochs=10)

    return model

# Evaluate
def evaluate_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_acc*100.0}')

# Predict
def predict(model, X):
    y_prob = model.predict(X)
    y_pred = tf.argmax(y_prob, axis=1).numpy()
    return y_pred

# Real-time audio analysis
def process_mic(indata, frames, time, status, model):
    if status:
        print(status)

    # Mel spectrogram extraction
    melspec = librosa.feature.melspectrogram(indata[:, 0])
    db_mel = librosa.power_to_db(melspec)

    # Make prediction
    features = db_mel.reshape(1, *db_mel.shape)
    prediction = model.predict(features)

    # Check prediction confidence
    pred_class = np.argmax(prediction)
    prob = np.max(prediction)

    if prob > 0.5:
        print(f"Recognized song as {genres[pred_class]}")

def real_time_audio_analysis(model):
    print("Press Ctrl+C to stop the real-time analysis.")
    with sd.InputStream(callback=lambda indata, frames, time, status: process_mic(indata, frames, time, status, model)):
        sd.sleep(1000000)  # Adjust this value based on the desired duration of real-time analysis

# Usage:
X, y = load_data()
X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

X_train, y_train = preprocess_data(X_train, y_train)
X_val, y_val = preprocess_data(X_val, y_val)
X_test, y_test = preprocess_data(X_test, y_test)

input_shape = X_train.shape[1:]
n_classes = y_train.shape[1]

model = build_model(input_shape, n_classes)
model = train_model(model, X_train, y_train, X_val, y_val)
evaluate_model(model, X_test, y_test)

# Run real-time audio analysis
real_time_audio_analysis(model)
