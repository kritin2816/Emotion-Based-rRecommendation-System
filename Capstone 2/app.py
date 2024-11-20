from flask import Flask, render_template, request
import mood_recommendation  # Import mood_recommendation.py directly

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    user_input = request.form['mood_input']
    
    # Check if user_input is provided
    if user_input:
        result = mood_recommendation.recommend_based_on_input(user_input)
        
        # Ensure the result is not None before unpacking
        if result is not None:
            detected_mood, recommended_songs = result
            return render_template('result1.html', mood=detected_mood, songs=recommended_songs)
        else:
            return "Could not detect mood or recommend songs. Please try again.", 500
    else:
        # If user input is empty or missing
        return "Mood input is required. Please enter your mood.", 400

if __name__ == '__main__':
    app.run(debug=True)
