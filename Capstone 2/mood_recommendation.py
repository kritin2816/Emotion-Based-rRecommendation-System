
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr

data = pd.read_csv(r'/Users/kritinreddy/Documents/Capstone final/data_moods.csv')
# Load the fine-tuned model and tokenizer
model_path = r'/Users/kritinreddy/Documents/Capstone final/fine_tuned_bert_model'

# Load the model (use BertModel instead of BertForSequenceClassification)
model = BertModel.from_pretrained(model_path)

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)


# Step 1: Define example sentences to represent each mood
mood_examples = {
    "Happy": [
        "I feel great today!", "I'm so happy and excited!", "This is a wonderful day!",
        "Everything is going so well!", "I'm having the best day ever!", "I can't stop smiling!",
        "I'm in a great mood!", "Feeling so positive and happy!", "Today is a day to celebrate!",
        "I feel on top of the world!", "Everything is working out for me!", "I'm thrilled about life!",
        "I feel so blessed!", "My heart is full of joy!", "I'm loving this moment!",
        "I just received great news!", "The sun is shining, and so am I!", "Life feels so good right now!",
        "I'm radiating happiness!", "This is a happy place!", "I love how today is going!",
        "I woke up feeling great!", "I'm smiling so much today!", "Everything seems to be falling into place.",
        "I love this vibe!", "Feeling so grateful right now!", "Today is the best!", "I’m feeling amazing.",
        "This day is full of positive energy.", "I’m so glad to be here.", "I’m feeling really joyful today!",
        "I can’t stop laughing!", "I’m so proud of how far I’ve come.", "I’m grateful for all of this.",
        "Today is the best day ever!", "My happiness knows no bounds.", "I just can't stop smiling today.",
        "I feel like dancing!", "I’m filled with positivity.", "I feel like I can achieve anything!",
        "My heart is full of joy!", "Everything is going according to plan.", "I’ve never felt better.",
        "I feel like the luckiest person in the world.", "I’m so excited for what’s to come.",
        "I’m bursting with excitement!", "The day is just full of good vibes.", "I feel unstoppable.",
        "I’m on top of the world today.", "I’m so happy I could jump for joy.", "Everything feels right today.",
        "I’m just so happy right now.", "I’m living my best life today!", "I am so thrilled!",
        "What a joyful moment!", "I feel so lucky and happy.", "I’m incredibly happy today!",
        "I’m feeling more energized than ever!", "Everything just feels right in the world.",
        "I’ve never been so happy!", "Today is filled with so much joy.", "I am in such a great mood!",
        "I feel like today is going to be amazing!", "I'm excited about what's next.", "I feel fantastic!",
        "Everything is going better than expected!", "I feel on cloud nine!", "I feel so full of life!",
        "I can’t help but feel happy right now.", "I’m looking forward to everything today.", "I'm so content!",
        "I’m so happy I’m speechless.", "I feel really blessed right now.", "I’m grinning from ear to ear!",
        "I’m in heaven right now.", "I'm feeling blessed.", "Life is just perfect right now!", "This is the life!",
        "I’ve never felt so grateful.", "I feel like the happiest person alive!", "I’m so lucky today!",
        "I'm having the best time of my life!", "I am so fortunate right now.", "Feeling happy and content.",
        "This is such a positive experience.", "I’m on top of the world!", "I love how everything is going.",
        "I'm just so happy with everything right now.", "The world feels right today!", "I feel like I can do anything!"
    ],
    "Sad": [
        "I'm feeling down.", "Today is not my day.", "I feel so sad.", "Everything feels so heavy.",
        "I'm just not feeling it today.", "I can't shake this sadness.", "I don't have the energy for anything.",
        "I'm having a tough time today.", "My heart feels heavy.", "I feel empty inside.",
        "Everything seems so dull right now.", "I just want to be alone.", "Nothing feels right today.",
        "I don't know what's wrong with me.", "I'm overwhelmed by sadness.", "I can't stop thinking about it.",
        "I feel like everything is falling apart.", "I'm not in a good place mentally.", "I feel stuck.",
        "I just want to escape.", "It's hard to be positive right now.", "I feel like crying.",
        "I'm feeling emotionally drained.", "I can't find joy in anything.", "I feel so lost.",
        "Nothing seems to make me happy.", "I can't focus on anything right now.", "I feel completely hopeless.",
        "I don't know what’s going on with me.", "I feel disconnected from everything.", "I feel so lonely.",
        "I'm struggling to keep it together.", "I feel like I’m sinking.", "I can't stop feeling sad.",
        "I feel like I'm drowning in my emotions.", "I don’t want to talk to anyone right now.", "Everything seems pointless.",
        "I'm too tired to care about anything.", "I feel emotionally exhausted.", "I'm just going through the motions.",
        "I feel numb to everything.", "I wish things were different.", "I feel broken inside.",
        "I just want everything to stop.", "I feel like I’m fading away.", "Nothing feels good anymore.",
        "I can't see the light at the end of the tunnel.", "I feel like giving up.", "I feel like I'm not enough.",
        "I'm in a bad mental space.", "I don't feel motivated at all.", "I can’t seem to shake this feeling.",
        "I feel like there’s no way out.", "I’m not looking forward to anything.", "I feel like everything’s too much.",
        "I’m losing hope.", "I feel so disconnected.", "I just want to be left alone.", "Everything feels too overwhelming.",
        "I can’t stop thinking about what went wrong.", "I feel out of place.", "I feel so misunderstood.",
        "I feel like no one cares.", "I can’t seem to find happiness anywhere.", "I feel like I’ve lost everything.",
        "I just want to cry and let it all out.", "I feel drained and empty inside.", "I feel like I’m stuck in a hole.",
        "I feel hopeless and tired.", "I can’t escape these feelings.", "I’m tired of pretending to be okay.",
        "I feel like I’m not in control of my emotions.", "I feel defeated.", "Everything seems to be falling apart.",
        "I’m so tired of everything.", "I feel completely alone in this.", "I don’t know how to make it stop.",
        "I feel emotionally broken.", "I can’t stop crying.", "I'm feeling lost and hopeless.", "Nothing seems to help."
    ],
    "Energetic": [
        "I am enthusiastic!", "Let's get things done!", "I'm feeling very energetic today.",
        "I'm full of energy and ready to go!", "Nothing can stop me today!", "I feel unstoppable!",
        "I have so much energy right now!", "I'm ready to take on the world!", "I'm pumped up!",
        "Let's make today amazing!", "I’m on fire today!", "I feel like I could conquer anything.",
        "Let's get moving!", "I'm bursting with energy!", "I can't sit still today!",
        "I’m all charged up!", "I'm ready to take on whatever comes my way!", "I’ve got so much motivation!",
        "I'm full of life and energy!", "Let's make it happen today!", "I feel like a powerhouse!",
        "I’m feeling energized!", "I’m ready to run a marathon!", "I feel like a superhero today!",
        "I’m firing on all cylinders!", "Nothing is slowing me down today.", "I feel totally alive!",
        "I’m ready to tackle anything that comes my way!", "My energy levels are through the roof!",
        "I feel like I can take on the world!", "I’ve got this in the bag!", "I’m charged up and ready to go!",
        "I’m so hyped up!", "I feel like I’ve got endless energy today.", "Let’s get this party started!",
        "I feel like dancing!", "I’m ready for anything!", "I feel so motivated to succeed!",
        "Today’s going to be a productive day!", "I’m filled with energy and creativity.", "Nothing can stop me now!",
        "I feel so driven!", "Let’s do this!", "I’m ready to take on the world!", "I feel so powerful!",
        "I’m ready for whatever comes my way!", "I’m in the mood to work hard!", "I’ve got more energy than I know what to do with!",
        "I can’t stop moving today!", "I’m full of ideas!", "I’m feeling the adrenaline!",
        "I feel like taking on new challenges.", "I’m pumped up for success!", "I’m feeling unstoppable today!",
        "I feel like I'm at my best right now.", "I’m excited to see where today takes me.", "I’m totally ready to go.",
        "I’m ready for anything the day throws at me.", "I'm moving at full speed today!", "I feel so charged up!",
        "I'm ready to crush my goals!", "Nothing can hold me back today!", "I’m so motivated and ready to win!",
        "I feel like I'm in my prime.", "I’m at the top of my game today!", "I can feel the energy flowing through me.",
        "I’m ready to take on the world with full force!", "I’m super motivated!"],
    "Calm": [
        "I'm calm and peaceful.", "Feeling very relaxed.", "It's a nice, quiet day.",
        "I feel centered and at peace.", "I’m in a tranquil state of mind.", "Everything feels serene.",
        "I’m just enjoying this peaceful moment.", "I feel very at ease.", "I’m in my zen place.",
        "It’s so quiet and calming.", "I feel grounded.", "I’m in a state of mindfulness.",
        "I feel balanced and calm.", "Everything is so still right now.", "I’m just breathing deeply and relaxing.",
        "I feel totally at peace.", "I’m taking it easy today.", "There’s a calmness in the air.",
        "I’m not rushing today, just going with the flow.", "I feel like everything is in perfect harmony.",
        "I’m enjoying the stillness around me.", "Everything is calm and steady.", "I’m feeling calm and collected.",
        "I’m enjoying the quiet moments.", "I’m in a peaceful state of mind.", "I’m at peace with myself.",
        "I’m feeling really grounded today.", "I’m completely relaxed.", "The day feels so serene.",
        "I’m in the mood for some relaxation.", "I feel at ease with everything around me.", "Everything seems so peaceful.",
        "I feel calm in the midst of chaos.", "I’m feeling the stillness around me.", "I feel a deep sense of calm.",
        "I’m not stressing about anything today.", "I’m just enjoying the calmness of the moment.", "There’s a gentle calmness in the air.",
        "I feel like everything is in its right place.", "I’m feeling light and unburdened.", "I’m taking a moment for myself.",
        "The world feels at peace around me.", "I’m feeling calm and tranquil.", "I’m in no rush today.",
        "I’m savoring the peace and quiet.", "I’m embracing this quiet moment.", "I’m enjoying the serenity around me.",
        "Everything feels so effortless today.", "I’m enjoying the calm energy in the air.", "I’m feeling calm and content.",
        "I’m enjoying this peaceful atmosphere.", "I’m at ease with everything happening around me.", "I feel so relaxed and balanced.",
        "I’m embracing the calmness of this moment.", "I’m not letting anything disturb my peace.", "Everything feels light and peaceful.",
        "I’m feeling so serene right now.", "I’m feeling completely at ease.", "I’m enjoying this quiet time.",
        "I’m feeling so relaxed and calm.", "I feel like I’m floating on air.", "I’m in a peaceful, relaxed state.",
        "There’s a soothing calmness around me.", "I’m feeling calm and unhurried.", "Everything is moving at the perfect pace.",
        "I’m really at peace with the situation.", "I’m soaking in the tranquility of the moment.", "I’m feeling calm and clear-headed.",
        "I’m in a slow and peaceful mood today.", "I’m at ease with my surroundings.", "The stillness is so calming.",
        "I’m feeling the serenity of the moment.", "I’m enjoying the calm flow of things.", "I feel completely at peace with myself.",
        "Everything feels balanced and peaceful.", "I’m in the mood for some quiet reflection.", "I’m letting everything unfold naturally.",
        "I’m feeling calm despite everything happening around me.", "I’m savoring the quiet moments today.", "I feel the peace within me.",
        "I’m embracing the stillness and quiet.", "I feel so relaxed and at peace with my surroundings.", "I’m content with the way things are going.",
        "I’m enjoying the calm of nature today.", "I feel like everything is in a peaceful rhythm.", "I’m in the mood to be still and quiet.",
        "I’m feeling so centered today.", "Everything feels perfect just the way it is.", "I feel calm even when things are busy.",
        "I’m taking things one step at a time.", "I’m enjoying this quiet, peaceful moment.", "I feel a deep sense of tranquility today.",
        "I’m feeling peaceful and calm inside.", "I’m at peace with everything happening in my life.", "I’m finding peace in the little things.",
        "I’m feeling so composed and centered.", "Everything feels so simple and peaceful.", "I’m feeling calm and untroubled today.",
        "I’m at peace with where I am in life.", "I feel like I’m in a peaceful bubble.", "I’m letting go of all the tension.",
        "I’m in a peaceful, relaxed state of mind.", "I’m feeling calm and in control of my emotions.", "I’m allowing myself to relax and unwind.",
        "I’m embracing the peaceful flow of today.", "I’m feeling calm and at ease with my thoughts.", "I’m grateful for this peaceful moment.",
        "I’m centered and at ease with everything.", "I’m taking everything in stride today.", "I feel calm and ready for whatever comes.",
        "I’m enjoying this peaceful rhythm in my life.", "I’m feeling calm, even in the midst of chaos.", "I’m at peace with my present situation.",
        "I’m soaking in the peaceful moments of today.", "I’m feeling calm in the face of uncertainty.", "I feel a quiet peace inside me."
    ]
}


# Function to generate embeddings for songs
def get_song_embeddings(dataset, tokenizer, model):
    model.eval()
    embeddings = []
    for _, song in dataset.iterrows():
        text = song['name'] + " " + song['artist'] + " " + song['album']
        inputs = tokenizer(text, max_length=128, padding='max_length', truncation=True, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        song_embedding = outputs.last_hidden_state.mean(dim=1).numpy()
        embeddings.append(song_embedding)

    return np.vstack(embeddings)

# Generate embeddings for all songs
song_embeddings = get_song_embeddings(data, tokenizer, model)

# Generate average embeddings for each mood
def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

mood_embeddings = {}
for mood, examples in mood_examples.items():
    example_embeddings = [get_sentence_embedding(sentence) for sentence in examples]
    mood_embeddings[mood] = np.mean(example_embeddings, axis=0)

# Function to detect mood based on user input
def detect_mood(user_input):
    user_embedding = get_sentence_embedding(user_input)
    similarities = {mood: cosine_similarity([user_embedding], [embedding])[0][0] for mood, embedding in mood_embeddings.items()}
    detected_mood = max(similarities, key=similarities.get)
    return detected_mood

# Function to recommend songs based on detected mood
def recommend_songs(detected_mood, data, song_embeddings, top_k=5):
    mood_songs = data[data['mood'] == detected_mood]

    if mood_songs.empty:
        return []

    mood_indices = mood_songs.index
    mood_embeddings = song_embeddings[mood_indices]

    similarity_matrix = cosine_similarity(mood_embeddings)
    recommendations = []

    for idx in range(len(mood_indices)):
        similar_indices = similarity_matrix[idx].argsort()[::-1][1:top_k+1]
        similar_songs = mood_songs.iloc[similar_indices]
        recommendations.extend(similar_songs[['name', 'artist', 'album', 'mood']].to_dict('records'))

    return recommendations

# New function that integrates mood detection and song recommendation
def recommend_based_on_input(user_input):
    detected_mood = detect_mood(user_input)
    recommended_songs = recommend_songs(detected_mood, data, song_embeddings, top_k=5)
    
    if detected_mood and recommended_songs:
        return detected_mood, recommended_songs
    return None

# Voice Recognition and Mood Detection from Voice Input
def voice_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        user_input = recognizer.recognize_google(audio)
        return user_input
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        return ""

# Function to integrate voice input into the mood detection and recommendation system
def recommend_songs_from_voice():
    user_input = voice_to_text()
    if user_input:
        detected_mood, recommended_songs = recommend_based_on_input(user_input)
        if detected_mood and recommended_songs:
            return detected_mood, recommended_songs
        return None
    return None

