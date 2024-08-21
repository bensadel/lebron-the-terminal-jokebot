import random  # For generating random choices for responses
import json  # For handling JSON data
import pickle  # For loading pre-trained models and data
import numpy as np  # For numerical operations on arrays

import nltk  # For natural language processing tasks
from nltk.stem import WordNetLemmatizer  # For word lemmatization (reducing words to their base form)

from tensorflow.keras.models import load_model  # For loading pre-trained Keras models

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the intents from a JSON file that defines possible user intents and bot responses
intents = json.loads(open("intents(funny).json").read())

# Load the pre-processed words and classes (intents) from pickled files
words = pickle.load(open("words(output).pkl", "rb"))
classes = pickle.load(open("classes(output).pkl", "rb"))

# Load the pre-trained chatbot model
model = load_model("chatbotmodel.h5")


# Function to tokenize and lemmatize the input sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)  # Tokenize the sentence into words
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]  # Lemmatize each word
    return sentence_words  # Return the list of cleaned-up words


# Function to convert a sentence into a bag-of-words array
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)  # Clean up the input sentence
    bag = [0] * len(words)  # Initialize a bag of words with zeroes
    for w in sentence_words:
        for i, word in enumerate(words):  # Match words with the vocabulary
            if word == w:
                bag[i] = 1  # Set the corresponding position to 1 if the word is present
    return np.array(bag)  # Return the bag-of-words as a NumPy array


# Function to predict the class (intent) of the input sentence
def predict_class(sentence):
    bow = bag_of_words(sentence)  # Convert the sentence into a bag-of-words
    res = model.predict(np.array([bow]))[0]  # Predict the intent using the model
    ERROR_THRESHOLD = 0.25  # Set a threshold for filtering out low-confidence predictions
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]  # Filter predictions above the threshold
    results.sort(key=lambda x: x[1], reverse=True)  # Sort by probability in descending order
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})  # Return intents with probabilities
    return return_list  # Return the list of predicted intents


# Function to generate a response based on the predicted intent
def get_response(intents_list, intents_json):
    if intents_list: 
        tag = intents_list[0]["intent"]  # Get the most probable intent
        list_of_intents = intents_json["intents"]  # Get the list of all intents

        valid_intents = [intent["tags"] for intent in list_of_intents]  # Extract all valid tags
        if tag in valid_intents:
            for i in list_of_intents:
                if i["tags"] == tag:
                    result = random.choice(i["responses"])  # Choose a random response for the intent
                    break
        else:
            result = "Invalid input! Please try again."  # Handle invalid intents
    else:
        result = "I'm not sure what you're asking. Please try again."  # Handle no matches
    return result  # Return the response


# Print a message to indicate the bot is running
print("TYPE! BOT IS RUNNING!")

# Continuous loop to keep the bot running
while True:
    message = input("")  # Get input from the user
    ints = predict_class(message)  # Predict the intent of the input
    res = get_response(ints, intents)  # Generate a response based on the predicted intent
    print(res)  # Print the bot's response
