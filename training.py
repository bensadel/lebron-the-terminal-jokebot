import random  # For randomizing the training data
import json  # For loading and handling JSON data
import pickle  # For saving and loading pre-processed data
import numpy as np  # For numerical operations on arrays

import nltk  # For natural language processing tasks
from nltk.stem import WordNetLemmatizer  # For word lemmatization

from tensorflow.keras.models import Sequential  # For creating a sequential neural network model
from tensorflow.keras.layers import Dense, Activation, Dropout  # For adding layers to the neural network
from tensorflow.keras.optimizers import SGD  # For stochastic gradient descent optimizer

# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the intents from the JSON file which defines user intents and bot responses
intents = json.loads(open("intents(funny).json").read())

# Initialize lists to hold words, classes (intents), and documents (patterns and associated intents)
words = []
classes = []
documents = []
ignore_letters = ["?", "!", "-", ",", ";", ":"]  # Characters to ignore during tokenization

# Loop through each intent in the JSON file
for intent in intents["intents"]:
    for pattern in intent["patterns"]:  # Loop through each pattern (example sentence) in the intent
        word_list = nltk.word_tokenize(pattern)  # Tokenize the pattern into words
        words.extend(word_list)  # Add the words to the words list
        documents.append((word_list, intent["tags"]))  # Add the pattern and associated intent to documents
        if intent["tags"] not in classes:
            classes.append(intent["tags"])  # Add the intent to the classes list if not already present

print("Done 1")  # Indicate completion of the first stage

# Lemmatize and sort the words, removing any ignored characters
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

# Sort the classes (intents)
classes = sorted(set(classes))

# Save the words and classes lists to pickle files for later use
pickle.dump(words, open("words(output).pkl", "wb"))
pickle.dump(classes, open("classes(output).pkl", "wb"))

# Initialize training data and an empty output vector
training = []
output_empty = [0] * len(classes)

# Create the training data by converting each pattern into a bag-of-words model
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]  # Lemmatize the words in the pattern
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)  # Create a binary array indicating word presence

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1  # Mark the appropriate intent in the output vector
    training.append([bag, output_row])  # Add the bag-of-words and output vector to the training data

# Shuffle the training data to ensure randomness
random.shuffle(training)

# Convert the training data into numpy arrays
training = np.array(training)
train_x = list(training[:, 0])  # Features (bag-of-words)
train_y = list(training[:, 1])  # Labels (intents)

print("Done 2")  # Indicate completion of the second stage

# Build the Sequential model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))  # First hidden layer with 128 neurons
model.add(Dropout(0.5))  # Dropout layer to prevent overfitting
model.add(Dense(64, activation="relu"))  # Second hidden layer with 64 neurons
model.add(Dropout(0.5))  # Another dropout layer
model.add(Dense(len(train_y[0]), activation="softmax"))  # Output layer with a neuron for each intent

# Set up the optimizer with learning rate, momentum, and Nesterov acceleration
sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)

# Compile the model with categorical cross-entropy loss function and accuracy as a metric
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Train the model on the training data for 200 epochs with a batch size of 5
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save the trained model to a file for later use
model.save("chatbotmodel.h5", hist)

print("Done 3")  # Indicate completion of the third stage
