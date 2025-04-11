from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import pickle
import tensorflow as tf
import gensim.downloader as api
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import os

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load Word2Vec model (Google News - reduced for performance)
print("Loading Word2Vec model...")
model_path = api.load("word2vec-google-news-300", return_path=True)
word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True, limit=50000)

# Load dataset
try:
    # Use error_bad_lines=False to skip problematic rows and warn about them
    df = pd.read_csv("../stdata.csv", 
                     encoding='utf-8', 
                     on_bad_lines='skip',
                     engine='python',  # Use python engine for better error handling
                     quotechar='"',    # Handle quoted fields properly
                     escapechar='\\')  # Handle escape characters
    
    print(f"Successfully loaded dataset with {len(df)} rows.")
    
    # Make column names consistent (lowercase)
    df.columns = df.columns.str.strip().str.lower()
    
    # Check for required columns
    required_columns = {"question", "desired_answer"}
    missing_columns = required_columns - set(df.columns)
    
    if missing_columns:
        # If columns are missing but have similar names, try to map them
        column_mapping = {}
        for col in df.columns:
            for req_col in missing_columns:
                if req_col in col.lower() or col.lower() in req_col:
                    column_mapping[col] = req_col
                    print(f"Mapping column '{col}' to required column '{req_col}'")
        
        # Rename columns based on mapping
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        # Check again after mapping
        still_missing = required_columns - set(df.columns)
        if still_missing:
            print(f"Warning: Could not find required columns: {still_missing}")
            print(f"Available columns: {list(df.columns)}")
            # Use default column names if they exist with different capitalization
            for req_col in still_missing:
                candidates = [c for c in df.columns if c.lower() == req_col.lower()]
                if candidates:
                    df = df.rename(columns={candidates[0]: req_col})
                    print(f"Using '{candidates[0]}' as '{req_col}'")
    
    # Final check for required columns
    missing = required_columns - set(df.columns)
    if missing:
        raise KeyError(f"Required columns {missing} are missing!")
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Creating empty dataframe with required columns...")
    df = pd.DataFrame(columns=["question", "desired_answer"])

# Load trained LSTM model
print("Loading LSTM model...")
custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}  # Fix MSE loss loading
model = tf.keras.models.load_model("../sae.keras", custom_objects=custom_objects)

# Load the tokenizer
try:
    with open("../tokenizer1.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)

    if not isinstance(tokenizer, Tokenizer):
        print("❌ Tokenizer is corrupted! Re-saving it...")
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(["This is a test sentence"])
        with open("../tokenizer1.pkl", "wb") as handle:
            pickle.dump(tokenizer, handle)
        print("✅ New tokenizer.pkl saved. Try running app.py again!")
except Exception as e:
    print("Tokenizer not found. Using default.")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(["This is a test sentence"])
    with open("../tokenizer1.pkl", "wb") as handle:
        pickle.dump(tokenizer, handle)

MAX_LENGTH = 50

# NLP tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalnum() and w not in stop_words]
    return " ".join(tokens)

def compute_wmd(text1, text2):
    text1_tokens = preprocess_text(text1).split()
    text2_tokens = preprocess_text(text2).split()
    if not text1_tokens or not text2_tokens:
        return float("inf")
    try:
        wmd = word_vectors.wmdistance(text1_tokens, text2_tokens)
        return max(0, min(wmd, 2))  # Normalize WMD score
    except Exception as e:
        print(f"❌ WMD Error: {e}")
        return 2

# Train TF-IDF vectorizer
vectorizer = TfidfVectorizer()
if not df.empty:
    vectorizer.fit(df["desired_answer"].dropna().tolist())

def compute_cosine_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    try:
        text_vectors = vectorizer.transform([text1, text2])
        return max(0, min(cosine_similarity(text_vectors[0], text_vectors[1])[0][0], 1))
    except Exception as e:
        print(f"❌ Cosine Similarity Error: {e}")
        return 0.0

@app.route('/favicon.ico')
def favicon():
    return "", 204  # Return empty response with "No Content" status

def get_random_question():
    try:
        if df.empty or "question" not in df.columns or "desired_answer" not in df.columns:
            return "What is the purpose of a compiler in programming?", "A compiler translates high-level programming code into machine code that can be executed by the computer."
        
        # Sample a complete row (no NaN values in important columns)
        valid_rows = df.dropna(subset=["question", "desired_answer"])
        
        if valid_rows.empty:
            return "What is the time complexity of Binary Search algorithm?", "The time complexity is O(log n) because it reduces the problem size by half in each step."
        
        row = valid_rows.sample(n=1).iloc[0]
        return row["question"], row["desired_answer"]
    except Exception as e:
        print(f"Error in get_random_question: {e}")
        return "What is the purpose of a cache memory in computer architecture?", "Cache memory stores frequently accessed data to provide faster access than main memory."

@app.route('/')
def index():
    session.clear()
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/demo', methods=['GET', 'POST'])
def demo():
    """Handles the demo page for evaluating answers."""
    
    # Check if the Next Question button was clicked
    if request.method == 'POST' and request.form.get('next_question'):
        # Get a new random question
        question, desired_answer = get_random_question()
        session['question'] = question
        session['desired_answer'] = desired_answer
        return render_template('demo.html', question=question, desired_answer=desired_answer)
    
    if 'question' not in session or 'desired_answer' not in session:
        question, desired_answer = get_random_question()
        session['question'] = question
        session['desired_answer'] = desired_answer
    else:
        question = session['question']
        desired_answer = session['desired_answer']

    student_answer = ""
    result = None

    if request.method == 'POST':
        try:
            student_answer = request.form.get('student_answer', "")
            
            # Handle empty student answer
            if not student_answer.strip():
                result = {
                    'score': 0,
                    'cosine_similarity': 0,
                    'wmd_score': 0,
                    'feedback': "You didn't provide an answer. Please try again."
                }
                return render_template('demo.html', question=question, desired_answer=desired_answer, 
                                      student_answer=student_answer, result=result)
            
            # Handle exact match with desired answer
            if student_answer.strip().lower() == desired_answer.strip().lower():
                result = {
                    'score': 5,
                    'cosine_similarity': 1.0,
                    'wmd_score': 0,
                    'feedback': "Perfect! Your answer exactly matches the expected response."
                }
                return render_template('demo.html', question=question, desired_answer=desired_answer, 
                                      student_answer=student_answer, result=result)
            
            # Preprocess both answers
            student_processed = preprocess_text(student_answer)
            desired_processed = preprocess_text(desired_answer)

            # Calculate similarity metrics
            wmd_score = compute_wmd(student_processed, desired_processed)
            cosine_sim = compute_cosine_similarity(student_processed, desired_processed)

            # Prepare for model prediction
            seq = tokenizer.texts_to_sequences([student_processed])
            padded_seq = pad_sequences(seq, maxlen=MAX_LENGTH)

            try:
                # Predict score using trained LSTM model
                predicted_score = model.predict(padded_seq, verbose=0)[0][0]
                # Normalize predicted score to 0-1 range if needed
                predicted_score = max(0, min(predicted_score, 1))  # Clamp between 0 and 1
            except Exception as model_error:
                print(f"Model prediction error: {model_error}")
                # Fallback prediction based only on similarity metrics
                predicted_score = 0.5  # Default mid-range value

            # Normalize similarity scores to 0-5
            wmd_normalized = (1 - min(wmd_score / 2, 1)) * 5
            cosine_normalized = cosine_sim * 5

            # Combine scores
            final_score = (0.6 * predicted_score * 5) + (0.4 * (wmd_normalized + cosine_normalized) / 2)
            final_score = max(0, min(5, final_score))  # Ensure within 0-5

            print("Student Answer Processed:", student_processed)
            print("Desired Answer Processed:", desired_processed)
            print("WMD Score:", wmd_score)
            print("Cosine Similarity:", cosine_sim)
            print("LSTM Predicted Score:", predicted_score)
            print("Final Score (out of 5):", final_score)

            result = {
                'score': round(final_score, 2),
                'cosine_similarity': round(cosine_sim, 2),
                'wmd_score': round(wmd_score, 2),
                'feedback': generate_feedback(final_score)
            }
        except Exception as e:
            print(f"Error in demo processing: {e}")
            result = {
                'score': 0,
                'cosine_similarity': 0,
                'wmd_score': 0,
                'feedback': "An error occurred while processing your answer. Please try again."
            }

    return render_template('demo.html', question=question, desired_answer=desired_answer, student_answer=student_answer,
                         result=result)

def generate_feedback(score):
    if score >= 4.5:
        return "Excellent! Your answer closely matches the expected response."
    elif score >= 3.5:
        return "Good job! Your answer covers most of the key points."
    elif score >= 2.5:
        return "Satisfactory. Your answer includes some key points but misses others."
    elif score >= 1.5:
        return "Needs improvement. Your answer is missing many key points."
    else:
        return "Your answer differs significantly from the expected response. Please review the material."

@app.route('/game', methods=['GET', 'POST'])
def game():
    """Game page that allows users to answer questions in a quiz-like format."""
    
    # Initialize game state
    if 'game_questions' not in session:
        # Get 5 random questions for the game
        game_questions = []
        for _ in range(5):
            q, a = get_random_question()
            game_questions.append({'question': q, 'desired_answer': a})
        session['game_questions'] = game_questions
        session['current_question_idx'] = 0
        session['total_score'] = 0
        session['answered_current'] = False
    
    # Get current game state
    game_questions = session['game_questions']
    current_idx = session['current_question_idx']
    total_score = session['total_score']
    total_questions = len(game_questions)
    
    # Check if game is completed
    if current_idx >= total_questions:
        return render_template('game.html', 
                              game_completed=True,
                              total_score=total_score,
                              total_questions=total_questions,
                              current_question_num=total_questions)
    
    # Get current question
    current_question = game_questions[current_idx]
    question = current_question['question']
    desired_answer = current_question['desired_answer']
    
    # Process POST requests
    student_answer = ""
    result = None
    
    if request.method == 'POST':
        # Move to next question if requested and allowed
        if request.form.get('next_question') and session.get('answered_current'):
            session['current_question_idx'] = current_idx + 1
            session['answered_current'] = False
            # Return to the beginning of the route to handle the next question
            return redirect(url_for('game'))
        
        # Process answer submission
        if request.form.get('submit_answer'):
            student_answer = request.form.get('student_answer', "")
            
            # Handle empty answer
            if not student_answer.strip():
                result = {
                    'score': 0,
                    'feedback': "You didn't provide an answer. Please try again."
                }
            else:
                # Use the same evaluation logic as the demo route
                student_processed = preprocess_text(student_answer)
                desired_processed = preprocess_text(desired_answer)
                
                # Calculate similarity metrics
                wmd_score = compute_wmd(student_processed, desired_processed)
                cosine_sim = compute_cosine_similarity(student_processed, desired_processed)
                
                try:
                    # Predict score using trained model
                    seq = tokenizer.texts_to_sequences([student_processed])
                    padded_seq = pad_sequences(seq, maxlen=MAX_LENGTH)
                    predicted_score = model.predict(padded_seq, verbose=0)[0][0]
                    predicted_score = max(0, min(predicted_score, 1))
                except Exception:
                    predicted_score = 0.5
                
                # Normalize and combine scores
                wmd_normalized = (1 - min(wmd_score / 2, 1)) * 5
                cosine_normalized = cosine_sim * 5
                final_score = (0.6 * predicted_score * 5) + (0.4 * (wmd_normalized + cosine_normalized) / 2)
                final_score = max(0, min(5, final_score))
                
                # Update total score
                session['total_score'] = total_score + final_score
                session['answered_current'] = True
                
                result = {
                    'score': round(final_score, 2),
                    'feedback': generate_feedback(final_score)
                }
    
    return render_template('game.html',
                          question=question,
                          desired_answer=desired_answer,
                          student_answer=student_answer,
                          result=result,
                          current_question_num=current_idx + 1,
                          total_questions=total_questions,
                          total_score=round(total_score, 2),
                          game_completed=False)

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    """Handle CSV file upload for custom questions."""
    global df
    
    if 'csv_file' not in request.files:
        return redirect(url_for('game'))
    
    file = request.files['csv_file']
    
    if file.filename == '':
        return redirect(url_for('game'))
    
    if file and file.filename.endswith('.csv'):
        try:
            # Save the uploaded file temporarily
            file_path = os.path.join(os.path.dirname(__file__), 'temp_upload.csv')
            file.save(file_path)
            
            # Read the uploaded CSV
            temp_df = pd.read_csv(file_path, 
                                 encoding='utf-8',
                                 on_bad_lines='skip',
                                 engine='python')
            
            # Make column names consistent
            temp_df.columns = temp_df.columns.str.strip().str.lower()
            
            # Check for required columns
            required_columns = {"question", "desired_answer"}
            missing_columns = required_columns - set(temp_df.columns)
            
            if missing_columns:
                # Try to find similar column names
                column_mapping = {}
                for col in temp_df.columns:
                    for req_col in missing_columns:
                        if req_col in col.lower() or col.lower() in req_col:
                            column_mapping[col] = req_col
                
                # Rename columns based on mapping
                if column_mapping:
                    temp_df = temp_df.rename(columns=column_mapping)
                
                # Check again after mapping
                still_missing = required_columns - set(temp_df.columns)
                if still_missing:
                    # Clean up and return to game
                    os.remove(file_path)
                    return redirect(url_for('game'))
            
            # Replace the global dataframe with the uploaded one if valid
            df = temp_df
            
            # Update the vectorizer with new data
            if not df.empty:
                global vectorizer
                vectorizer = TfidfVectorizer()
                vectorizer.fit(df["desired_answer"].dropna().tolist())
            
            # Clean up temp file
            os.remove(file_path)
            
            # Reset the game with new questions
            session.pop('game_questions', None)
            
        except Exception as e:
            print(f"CSV upload error: {e}")
    
    return redirect(url_for('game'))

if __name__ == '__main__':
    app.run(debug=True)
