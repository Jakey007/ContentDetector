# Import the necessary libraries and packages for OS logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Removing tensorflow messages to only output errors
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Importing the necessary libraries and packages for textual analysis
from tensorflow.keras.models import load_model
import pickle
import sys

# Load the previous classifier for content predictions
with open('tfidf_tokenizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Creating a function to make a prediction on a text
def predict_text(model, text):

    # Converts text to list for tokenization
    if isinstance(text, str):
        text = [text]

    # Make predictions on the inputted text
    X_new = tfidf_vectorizer.transform(text)  
    probabilities = model.predict(X_new) 

    class_labels = ['AI Generated', 'Human Content']
    threshold = 0.5

    # Convert probabilities to class labels based on threshold
    predicted_labels = (probabilities > threshold).astype(int)
    predicted_labels = predicted_labels.flatten()
    predicted_labels = [class_labels[label] for label in predicted_labels]

    return predicted_labels

# User passes the path to a txt file to have a prediction made
if __name__ == "__main__":
    
    # Outputs the usage of the script
    if len(sys.argv) != 2:
        print("Usage: python predict_script.py <text_file_path>")
        sys.exit(1)

    # Predefined Model Path
    model_path = 'text_classifier.keras'

    # Load the saved model
    model = load_model(model_path)

    # Read text from the specified file path
    text_file_path = sys.argv[1]
    with open(text_file_path, 'r', encoding='utf-8', errors='ignore') as file:
        text = file.read().strip()

    # Make prediction and output results
    prediction = predict_text(model, text)
    print('\nBased on Your Submission:\n', text)
    print("\nThis Text is Likely:", prediction, '\n')