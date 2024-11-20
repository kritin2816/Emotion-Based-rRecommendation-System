import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch import nn
import torch

# Load the CSV file containing the songs data
dat_mood = pd.read_csv(r'C:\Users\ramsh\OneDrive\Desktop\Capstone\data_moods.csv')  # Update this path as needed

# Load your fine-tuned BERT model for sequence classification
model_path = r'C:\Users\ramsh\OneDrive\Desktop\Capstone\fine_tuned_bert_model'
model = BertForSequenceClassification.from_pretrained(model_path)  # Use BertForSequenceClassification
tokenizer = BertTokenizer.from_pretrained(model_path)

# Define a function to detect mood from the user input using BERT model
def detect_mood_bert(user_input):
    # Tokenize the input sentence
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Run the model to get the prediction (logits)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract the logits and apply softmax to get probabilities
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    
    # Map the predicted class to a mood
    if predicted_class == 4:
        return "Happy"
    elif predicted_class == 3:
        return "Energetic"
    elif predicted_class == 2:
        return "Neutral"
    elif predicted_class == 1:
        return "Sad"
    else:
        return "Calm"

# Function to recommend songs based on detected mood
def recommend_based_on_input(user_input):
    # Detect the mood based on user input using BERT
    detected_mood = detect_mood_bert(user_input)
    print(f"Detected Mood: {detected_mood}")
    
    # Recommend songs based on the detected mood
    recommended_songs = dat_mood[dat_mood['mood'] == detected_mood]
    
    if recommended_songs.empty:
        print("Sorry, no songs were found for this mood.")
    else:
        print(f"Recommended Songs based on your mood ({detected_mood}):")
        print(recommended_songs[['name', 'artist', 'album']])

# Main function to interact with the user
if __name__ == "__main__":
    # Ask user to input their mood description as a sentence
    user_input = input("How are you feeling today? Describe your mood: ")
    recommend_based_on_input(user_input)
