from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import firebase_admin
from firebase_admin import credentials, firestore
import google.generativeai as genai
import mediapipe as mp
import speech_recognition as sr
import random
import cv2
import numpy as np
import os
from gtts import gTTS
from PIL import Image
from playsound import playsound
import bcrypt
import re
from flask_socketio import SocketIO, emit
import secrets
from cvzone.HandTrackingModule import HandDetector
import time
import json
import base64



def load_question_bank():
    with open('questions.json', 'r') as file:
        return json.load(file)

question_bank = load_question_bank()

jokes=[
       "Why did the banana go to the doctor? Because it wasn’t peeling well!",
       "What do you call a dinosaur that is sleeping? A dino-snore!",
       "Why do cows wear bells? Because their horns don’t work!",
       "What do you call a bear with no teeth? A gummy bear!",
       "Why did the teddy bear say no to dessert? Because it was stuffed!",
       "What’s a cat’s favorite color? Purr-ple!",
       "Why did the bicycle fall over? Because it was two-tired!",
       "What do you call a train that sneezes? Achoo-choo train!",
       "What’s a frog’s favorite candy? Lollihops!",
       "What do you call cheese that isn’t yours? Nacho cheese!",
       ]
facts = [
    "Butterflies taste with their feet.",
    "A group of fish is called a school.",
    "The sun is a big star that gives us light and heat.",
    "Elephants use their trunks to drink water, like a straw.",
    "Kangaroos can hop really fast—faster than a car in the city!",
    "Bees make honey to eat in winter.",
    "A baby kangaroo is called a joey.",
    "There are more stars in the sky than grains of sand on the beach!",
    "Penguins can’t fly, but they are great swimmers.",
    "Some trees can live for thousands of years."
]

def is_thumbs_up(fingers, lmList):
    """
    Determines if the detected hand is showing a thumbs-up gesture.
    """
    if fingers and lmList:
        # Landmark positions for the thumb and index finger
        thumb_tip = lmList[4]  # Thumb tip
        thumb_base = lmList[2]  # Thumb base
        index_tip = lmList[8]  # Index finger tip

        # Check if thumb is up and index finger is folded
        if thumb_tip[1] < thumb_base[1] and index_tip[1] > thumb_base[1]:
            return True

    return False


def getHandInfo(img):
    detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.8, minTrackCon=0.5)
    hands, img = detector.findHands(img, draw=True, flipType=True)
    if hands:
        hand1 = hands[0]
        lmList = hand1["lmList"]
        fingers = detector.fingersUp(hand1)
        return fingers, lmList
    else:
        return None, None

def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:  # Index finger up
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, tuple(prev_pos), tuple(current_pos), (255, 0, 255), 10)
    elif fingers == [1, 1, 1, 1, 1]:  # All fingers up (reset canvas)
        canvas = np.zeros_like(canvas)
    return current_pos, canvas


def sendToAI(model, canvas, fingers):
    if fingers == [1, 0, 0, 0, 0]:  # thumb up
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this math problem", pil_image])
        return response.text
    return ''


def check_Thumb(img):
    detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.8, minTrackCon=0.5)
    hands, img = detector.findHands(img, draw=False, flipType=True)

    if hands:
        hand1 = hands[0]
        lmList = hand1["lmList"]
        fingers = detector.fingersUp(hand1)

        if fingers and lmList:
            # Landmark positions for the thumb and index finger
            thumb_tip = lmList[4]  # Thumb tip
            thumb_base = lmList[2]  # Thumb base
            index_tip = lmList[8]  # Index finger tip

            # Check if thumb is up and index finger is folded
            if thumb_tip[1] < thumb_base[1] < index_tip[1]:
                return True

    return False

def is_valid_email(email):
    """Check if the email is in a valid format."""
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_regex, email) is not None


# Initialize Flask app
app = Flask(__name__)
socketio = SocketIO(app)

# Initialize Firebase
cred = credentials.Certificate("firebase_credentials.json")  # Replace with your Firebase credentials file
firebase_admin.initialize_app(cred)
db = firestore.client()

colours = ['Red', 'Blue', 'Green', 'Pink', 'Black', 'Yellow', 'Orange', 'White', 'Purple', 'Brown']
score = 0
timeleft = 30
game_active = False



from pydub import AudioSegment
from pydub.playback import play

def text_to_speech(text):
    tts = gTTS(text)
    audio_path = "temp.mp3"
    tts.save(audio_path)

    # Convert to WAV
    wav_path = "temp.wav"
    AudioSegment.from_mp3(audio_path).export(wav_path, format="wav")

    # Play the audio
    play(AudioSegment.from_wav(wav_path))

    # Clean up temporary files
    os.remove(audio_path)
    os.remove(wav_path)



def take_voice_input():
    """Capture and process voice input."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source, phrase_time_limit=4)

    try:
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand."
    except sr.RequestError:
        return "Sorry, I couldn't understand."

# Mapping for alphabet images and descriptions
with open("alphabetimages.json", "r") as file:
    alphabet_mapping = json.load(file)



@app.route('/')
def about():
    return render_template('about.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['POST'])
def signup():
    data = request.form
    username = data.get('username')
    password = data.get('password')
    first_name = data.get('firstName')
    last_name = data.get('lastName')

    if not is_valid_email(username):
        return jsonify({'message': 'Invalid email format!'}), 400

    # Check if user exists
    user_ref = db.collection('users').document(username)
    if user_ref.get().exists:
        return jsonify({'message': 'User already exists!'}), 400

    # Hash the password
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    # Save user data with the hashed password
    user_ref.set({
        'password': hashed_password.decode('utf-8'),
        'first_name': first_name,
        'last_name': last_name
    })

    scores_ref = user_ref.collection('scores')
    for subject in ['english', 'math', 'gk', 'miscellaneous']:
        scores_ref.document(subject).set({'scores': [0] * 10, 'subject': subject})


    # Store user info in session
    session['username'] = username
    session['first_name'] = first_name
    session['last_name'] = last_name


    return jsonify({'message': 'Signup successful!'}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.form
    username = data.get('username')
    password = data.get('password')

    # Verify user credentials
    user_ref = db.collection('users').document(username)
    user = user_ref.get()
    if user.exists:
        stored_password = user.to_dict().get('password')
        # Check the password against the stored hash
        if bcrypt.checkpw(password.encode('utf-8'), stored_password.encode('utf-8')):
            # Store user info in session
            user_data = user.to_dict()
            session['username'] = username
            session['first_name'] = user_data.get('first_name')
            session['last_name'] = user_data.get('last_name')
            return jsonify({'message': 'Login successful!'}), 200

    return jsonify({'message': 'Invalid credentials!'}), 401

@app.route('/home')
def home():
    # Redirect to the login page if the user is not logged in
    if 'username' not in session:
        return redirect(url_for('index'))

    return render_template(
        'home.html',
        first_name=session.get('first_name'),
        last_name=session.get('last_name')
    )


@app.route('/quiz')
def quiz():
    if 'username' not in session:
        return redirect(url_for('index'))

    return render_template('quiz.html')
@app.route('/english')
def english():
    if 'username' not in session:
        return redirect(url_for('index'))

    return render_template('english.html')

@app.route('/quizenglish')
def quizenglish():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('quizenglish.html')

@app.route('/quizgk')
def quizgk():
    if 'username' not in session:
        return redirect(url_for('index'))

    return render_template('quizgk.html')

@app.route('/quizmath')
def quizmath():
    if 'username' not in session:
        return redirect(url_for('index'))

    return render_template('quizmath.html')

@app.route('/quizmisc')
def quizmisc():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('quizmisc.html')

@app.route('/miscellaneous')
def miscellaneous():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('miscellaneous.html')

@app.route('/math')
def math():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('math.html')

@app.route('/gk')
def gk():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('gk.html')

@app.route('/get_leaderboard/<subject>', methods=['GET'])
def get_leaderboard(subject):
    if 'username' not in session:
        return {'error': 'User not logged in'}, 401

    username = session['username']
    user_ref = db.collection('users').document(username)
    scores_ref = user_ref.collection('scores').document(subject)
    scores_doc = scores_ref.get()

    if not scores_doc.exists:
        return {'error': f"No scores found for subject {subject}"}, 404

    scores = scores_doc.to_dict().get('scores', [0] * 10)  # Get the scores list
    return jsonify(scores)

@app.route('/speak', methods=['POST'])
def speak():
    data = request.json
    text = data.get('text')

    if not text:
        return {'error': 'No text provided'}, 400

    text_to_speech(text)
    return {'message': 'Speech played successfully'}, 200



@app.route('/get_questions/<subject>', methods=['GET'])
def get_questions(subject):
    questions = question_bank.get(subject, [])
    if not questions:
        return jsonify({'error': 'No questions found for this subject'}), 404

    random_questions = random.sample(questions, min(10, len(questions)))
    return jsonify(random_questions)



@app.route('/get_questions/miscellaneous', methods=['GET'])
def get_miscellaneous_questions():
    # Combine all questions
    all_questions = question_bank.get('english', [])+ question_bank.get('math', [])+question_bank.get('gk', [])
    # Randomly select 10 questions
    selected_questions = random.sample(all_questions, min(10,len(question_bank)))
    return jsonify(selected_questions)

@app.route('/update_score/<subject>', methods=['POST'])
def update_score(subject):
    if 'username' not in session:
        return {'error': 'User not logged in'}, 401

    data = request.json
    new_score = data.get('score')

    if new_score is None:
        return {'error': 'No score provided'}, 400

    username = session['username']
    user_ref = db.collection('users').document(username)
    scores_ref = user_ref.collection('scores').document(subject)

    score_doc = scores_ref.get()
    if not score_doc.exists:
        return {'error': 'Subject not found'}, 400

    scores = score_doc.to_dict().get('scores', [0] * 10)
    scores.insert(0, new_score)  # Add new score at the start
    scores = scores[:10]  # Keep only the last 10 scores

    scores_ref.set({'scores': scores, 'subject': subject, 'username': username}, merge=True)
    return {'message': 'Score updated successfully'}, 200


@app.route('/chat')
def chat():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('bot.html')

@socketio.on('start_conversation')
def conversation_loop():

    global is_connected
    is_connected=True
    """Continuously handle voice commands and send updates to the client."""
    while is_connected:

        emit("bot_response", {"text": "Listening..."}, broadcast=True)
        input_text = take_voice_input()


        flag1 = True
        if input_text == "Sorry, I couldn't understand.":
            emit("bot_response", {"text": f"Recognizing..."}, broadcast=True)
            emit("bot_response", {"text": f"Bot: {input_text}"}, broadcast=True)
            text_to_speech(input_text)
            flag1 = False
        else:
            emit("bot_response", {"text": f"Recognizing..."}, broadcast=True)
            emit("bot_response", {"text": f"User: {input_text}"}, broadcast=True)


        if input_text.lower() in ["exit","bye","goodbye", "good bye"]:
            emit("bot_response", {"text": "Goodbye! Have a nice day!"}, broadcast=True)
            text_to_speech('goodbye! have a nice day!')
            emit("redirect", {"url": url_for('home')}, broadcast=True)
            break

        elif input_text.lower() in ["addition", "add", "plus"]:
            emit("bot_response", {"text": "Bot: Executed addition for action."}, broadcast=True)
            text_to_speech('Executed addition for action.')
            emit("redirect", {"url": url_for('addition')}, broadcast=True)
            break

        elif input_text.lower() in ["subtraction", "minus", "subtract"]:
            emit("bot_response", {"text": "Bot: Executed subtraction for action."}, broadcast=True)
            text_to_speech('Executed subtraction for action.')
            emit("redirect", {"url": url_for('subtraction')}, broadcast=True)
            break

        elif input_text.lower() in ["count", "counting", "numbers"]:
            emit("bot_response", {"text": "Bot: Executed count for action."}, broadcast=True)
            text_to_speech('Executed count for action.')
            emit("redirect", {"url": url_for('count')}, broadcast=True)
            break

        elif input_text.lower() in [ "virtual whiteboard", "virtual white board","white board", "whiteboard"]:
            emit("bot_response", {"text": "Bot: Executed virtual white board for action."}, broadcast=True)
            text_to_speech('Executed virtual white board for action.')
            emit("redirect", {"url": url_for('vwb')}, broadcast=True)
            break

        elif input_text.lower() in[ "multiplication", "multiply"]:
            emit("bot_response", {"text": "Bot: Executed multiplication for action."}, broadcast=True)
            text_to_speech('Executplied multiplication for action.')
            emit("redirect", {"url": url_for('multiplication')}, broadcast=True)
            break

        elif input_text.lower() in[ "tic-tac-toe" ,"zero kati" ,"zero and crosses" ,"0 and crosses","tic tac toe"]:
            emit("bot_response", {"text": "Bot: Executed tictactoe for action."}, broadcast=True)
            text_to_speech('Executed tictactoe for action.')
            emit("redirect", {"url": url_for('tictactoe')}, broadcast=True)
            break

        elif input_text.lower() in ["sit stand", "sit and stand"]:
            emit("bot_response", {"text": "Bot: Executed sit stand for action. "}, broadcast=True)
            text_to_speech('Executed sit stand for action. ')
            emit("redirect", {"url": url_for('sitstand_page')}, broadcast=True)
            break

        elif input_text.lower() in [ "in and out" , "in out"]:
            emit("bot_response", {"text": "Bot: Executed in and out for action."}, broadcast=True)
            text_to_speech('Executed in and out for action.')
            emit("redirect", {"url": url_for('inout_page')}, broadcast=True)
            break

        elif input_text.lower() in ["secret number", "guess the secret number", "number guessing","number guessing game"]:
            emit("bot_response", {"text": "Bot: Executed secret number for action."}, broadcast=True)
            text_to_speech('Executed secret number for action.')
            emit("redirect", {"url": url_for('secret_number_page')}, broadcast=True)
            break

        elif input_text.lower() in ["whack a mole","whack the mole", "whack-a-mole","walk a mole","walk amul"]:
            emit("bot_response", {"text": "Bot: Executed whack a mole for action. "}, broadcast=True)
            text_to_speech('Executed whack a mole for action.')
            emit("redirect", {"url": url_for('whackamole')}, broadcast=True)
            break
        elif input_text.lower() in ["spot the difference", "spot difference", "spot a difference"]:
            emit("bot_response", {"text": "Bot: Executed spot the difference for action. "}, broadcast=True)
            text_to_speech('Executed spot the difference for action.')
            emit("redirect", {"url": url_for('spotthedifference')}, broadcast=True)
            break

        elif input_text.lower() in ["guess the colour", "colour guess","colour guessing","colour guessing game"]:
            emit("bot_response", {"text": "Bot: Executed guess the color for action."}, broadcast=True)
            text_to_speech('Executed guess the color for action.')
            emit("redirect", {"url": url_for('guess_color_page')}, broadcast=True)
            break

        elif input_text.lower() in ["hello", "hi","hii", "hi there", "hello there"]:
            emit("bot_response", {"text": "Hi there! How can I help you today?"}, broadcast=True)
            text_to_speech('Hi there! How can I help you today?')
            flag1=False

        elif input_text.lower() in[ "how are you", "how you doing", "how are you doing?"]:
            emit("bot_response", {"text": "I'm doing great! How about you?"}, broadcast=True)
            text_to_speech("I'm doing great! How about you?")
            flag1 = False

        elif input_text.lower() in ["what is your name", "your name", "name", "your name is"]:
            emit("bot_response", {"text": "I'm your friendly assistant bot, Kira!"}, broadcast=True)
            text_to_speech("I'm your friendly assistant bot, Kira!")
            flag1 = False

        elif input_text.lower() in ["what can you do","you can do what"]:
            emit("bot_response", {"text": "I can chat with you, answer questions, and assist you!"}, broadcast=True)
            text_to_speech("I can chat with you, answer questions, and assist you!")
            flag1 = False

        elif input_text.lower() in ["tell me a joke", "tell a joke", "joke", "tell joke", "joke tell"]:
            joke=random.choice(jokes)
            emit("bot_response", {"text": f"Here is a joke. {joke}"}, broadcast=True)
            text_to_speech('here is a joke. ' + joke)
            flag1 = False

        elif input_text.lower() in ["good morning", "morning"]:
            emit("bot_response", {"text": "Good morning! I hope you have a fantastic day ahead!"}, broadcast=True)
            text_to_speech("Good morning! I hope you have a fantastic day ahead!")
            flag1 = False

        elif input_text.lower() in ["good night","night"]:
            emit("bot_response", {"text": "Good night! Sweet dreams!"}, broadcast=True)
            text_to_speech("Good night! Sweet dreams!")
            flag1 = False

        elif input_text.lower() in[ "thank you","thank you so much"]:
            emit("bot_response", {"text": "You're welcome! Happy to help!"}, broadcast=True)
            text_to_speech("You're welcome! Happy to help!")
            flag1 = False

        elif input_text.lower() in ["can you help me","need help", "help" ,"help me" ]:
            emit("bot_response", {"text": "Of course! I'm here to help you explore and learn with me! Here’s what I can do for you on this platform: \n\nPlay fun and educational games Just say the game name like 'tic tac toe' or 'spot the difference'. \n\nTest your knowledge with quizzes Just say 'quiz' to start. \n\nLearn using activities \n For alphabets: Say commands like 'a for', 'b for', etc. \nFor math: Say 'count', 'addition', 'subtraction', or 'multiplication'. \nFor a virtual whiteboard: Say 'virtual whiteboard'. \n\nyou can also Talk to me. \nI can answer some basic questions, tell you a joke, or share an interesting fact. \n\n Just say the command, and I'll do it for you!"}, broadcast=True)
            time.sleep(3)
            flag1 = False

        elif input_text.lower() in[ "tell me something interesting", "tell me a fact" ,"tell a fact", "tell an interesting fact","tell a interesting fact","tell interesting fact","interesting fact","something interesting","tell something interesting"]:
            fact = random.choice(facts)
            emit("bot_response", {"text": f"Here is an interesting fact. {fact}"}, broadcast=True)
            text_to_speech('here is an interesting fact '+fact)
            flag1 = False

        elif input_text.lower() in[ "can you sing", "sing for me", "sing" ,"do you sing"]:
            emit("bot_response", {"text": "La la la! I hope that sounded good!"}, broadcast=True)
            text_to_speech("La la la! I hope that sounded good!")
            flag1 = False


        elif input_text.lower() in[ "where are you from","you from where","where from","from where"]:
            emit("bot_response", {"text": "I'm from StudyBuddy! Your friendly virtual assistant."}, broadcast=True)
            text_to_speech("I'm from Study Buddy! Your friendly virtual assistant.")
            flag1 = False


        elif input_text.lower() in ["quiz", "quizzes"]:
            emit("bot_response", {"text": "Bot: Executed quiz for action."}, broadcast=True)
            text_to_speech('Executed quiz for action. ')
            emit("redirect", {"url": url_for('quiz')}, broadcast=True)
            break



        else:
            # Process the voice input
            flag2=False
            for letter in alphabet_mapping:
                if input_text.lower() == f"{letter} for":
                    flag1 = False
                    flag2=True
                    break
            if flag2:
                emit("bot_response", {"text": f"Bot: Executed '{letter.upper()} for action'. Opening camera view..."},
                     broadcast=True)
                emit("redirect", {"url": url_for('alphabet', letter=letter)}, broadcast=True)
                break
        if flag1:
            emit("bot_response", {"text": "Bot: Invalid command! \n\n To play fun and educational games Just say the game name like 'tic tac toe' or 'spot the difference'. \n\nTo test your knowledge with quizzes Just say 'quiz' to start. \n\nTo learn using activities \n For alphabets: Say commands like 'a for', 'b for', etc. \nFor math: Say 'count', 'addition', 'subtraction', or 'multiplication'. \nFor a virtual whiteboard: Say 'virtual whiteboard'. \n\nyou can also Talk to me. \nI can answer some basic questions, tell you a joke, or share an interesting fact. \n\n Just say the command, and I'll do it for you!"}, broadcast=True)
            time.sleep(3)

@app.route('/activity')
def activity():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('activity.html')

@app.route('/alphabet')
def alphabet():
    if 'username' not in session:
        return redirect(url_for('index'))
    letter = request.args.get('letter', '')
    return render_template('alphabet.html', letter=letter)

@socketio.on('start_camera_feed')
def start_camera_feed(data):
    """Stream the processed video feed with a custom overlay to the client."""
    mp_Hands = mp.solutions.hands
    hands = mp_Hands.Hands()
    letter = data.get('letter', '').strip()
    global is_connected
    is_connected=True

    try:
        options = alphabet_mapping[letter]
        selected_option = random.choice(options)
        speaktext = selected_option["description"]
        image_filename = selected_option["image"]
        image_path = os.path.join(os.getcwd(), image_filename)
        text_to_speech(speaktext)
        print(image_path)
        sample_image = cv2.imread(image_path)
        if sample_image is None:
            text_to_speech("sample image is none")
            emit('video_feed', b'', broadcast=True)  # Send empty feed if image is invalid
            return

        image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
        edges = cv2.dilate(cv2.Canny(thresh, 0, 255), None)
        cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
        mask = np.zeros((256, 256), np.uint8)
        masked = cv2.drawContours(mask, [cnt], -1, 255, -1)

        masked_inv = cv2.bitwise_not(masked)
        masked_inv_rgb = cv2.merge([masked_inv, masked_inv, masked_inv])
        face_cascade = cv2.CascadeClassifier(r"C:\Users\ASUS\Downloads\archive\haarcascade_frontalface_default.xml")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            text_to_speech("cap is not opened")
            emit('video_feed', b'', broadcast=True)  # Notify client of failure
            return

        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                text_to_speech("no ret")
                break

            if check_Thumb(img):
                cap.release()
                text_to_speech("Thumbs up detected! Redirecting.")
                emit('redirect', {"url": url_for('home')}, broadcast=True)
                break
            if is_connected == False:
                break


            RGB_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(RGB_image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            handList=[]

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    for idx, lm in enumerate(handLms.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        handList.append((cx, cy))

                cx = handList[8][0] - 128
                cy = handList[8][1] - 128
                hi, wi, c = img.shape

                if (cx <= (wi - 256)) and (cy <= (hi - 256)) and (cy > 0) and (cx > 0):
                    masked_inv_rgb = cv2.resize(masked_inv_rgb, (256, 256))
                    sample_image = cv2.resize(sample_image, (256, 256))
                    mask_boolean = masked_inv_rgb[:, :, 0] == 0
                    mask_rgb_boolean = np.stack([mask_boolean, mask_boolean, mask_boolean], axis=2)
                    img[cy:cy + 256, cx:cx + 256, :] = (img[cy:cy + 256, cx:cx + 256, :] * ~mask_rgb_boolean[0:256, 0:256, :] + (sample_image * mask_rgb_boolean)[0:256, 0:256, :])
            else:
                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        masked_inv_rgb = cv2.resize(masked_inv_rgb, (w, h))
                        sample_image = cv2.resize(sample_image, (w, h))
                        mask_boolean = masked_inv_rgb[:, :, 0] == 0
                        mask_rgb_boolean = np.stack([mask_boolean, mask_boolean, mask_boolean], axis=2)
                        img[y:y + h, x:x + w, :] = img[y:y + h, x:x + w, :] * ~mask_rgb_boolean[0:h, 0:w, :] + (sample_image * mask_rgb_boolean)[0:h, 0:w, :]

            # Encode frame as JPEG
            _, buffer = cv2.imencode('.png', img)
            frame = buffer.tobytes()

            # Emit the frame to the client
            emit('video_feed', frame, broadcast=True, binary=True)
            socketio.sleep(0.003)

    except KeyError:
        text_to_speech("Invalid letter")
        emit('video_feed', b'', broadcast=True)
    except Exception as e:
        text_to_speech(f"Error occurred: {str(e)}")
        emit('video_feed', b'', broadcast=True)
    finally:
        cv2.destroyAllWindows()

@socketio.on('disconnect')
def handle_disconnect():
    global is_connected
    print("Client disconnected")
    is_connected = False


@app.route('/addition')
def addition():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('addition.html')

@socketio.on('addition')
def addition():
    try:
        mp_hands = mp.solutions.hands
        cap = cv2.VideoCapture(0)
        global is_connected
        is_connected = True
        if not cap.isOpened():
            text_to_speech("cap is not opened")
            emit('video_feed', b'', broadcast=True)  # Notify client of failure
            return

        with mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5, ) as hands:

            while cap.isOpened():
                ret, image = cap.read()
                if not ret:
                    text_to_speech("no ret")
                    break

                if check_Thumb(image):
                    cap.release()
                    text_to_speech("Thumbs up detected! Redirecting.")
                    emit('redirect', {"url": url_for('home')}, broadcast=True)
                    break

                if is_connected == False:
                    break

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                fingerCount1 = 0
                fingerCount2 = 0

                if results.multi_hand_landmarks:

                    for hand_landmarks in results.multi_hand_landmarks:
                        # Get hand index to check label (left or right)
                        handIndex = results.multi_hand_landmarks.index(hand_landmarks)
                        handLabel = results.multi_handedness[handIndex].classification[0].label

                        # Set variable to keep landmarks positions (x and y)
                        handLandmarks = []

                        # Fill list with x and y positions of each landmark
                        for landmarks in hand_landmarks.landmark:
                            handLandmarks.append([landmarks.x, landmarks.y])

                        if handLabel == "Left":
                            if handLandmarks[4][0] > handLandmarks[3][0]:
                                fingerCount1 += 1
                            if handLandmarks[8][1] < handLandmarks[6][1]:  # Index finger
                                fingerCount1 += 1
                            if handLandmarks[12][1] < handLandmarks[10][1]:  # Middle finger
                                fingerCount1 += 1
                            if handLandmarks[16][1] < handLandmarks[14][1]:  # Ring finger
                                fingerCount1 += 1
                            if handLandmarks[20][1] < handLandmarks[18][1]:  # Pinky
                                fingerCount1 += 1

                        elif handLabel == "Right":
                            if handLandmarks[4][0] < handLandmarks[3][0]:
                                fingerCount2 += 1
                            if handLandmarks[8][1] < handLandmarks[6][1]:  # Index finger
                                fingerCount2 += 1
                            if handLandmarks[12][1] < handLandmarks[10][1]:  # Middle finger
                                fingerCount2 += 1
                            if handLandmarks[16][1] < handLandmarks[14][1]:  # Ring finger
                                fingerCount2 += 1
                            if handLandmarks[20][1] < handLandmarks[18][1]:  # Pinky
                                fingerCount2 += 1

                # Display finger count
                sum = fingerCount1 + fingerCount2
                disp = str(fingerCount1) + ' + ' + str(fingerCount2) + ' = ' + str(sum)
                cv2.putText(image, disp, (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)
                _, buffer = cv2.imencode('.png', image)
                frame = buffer.tobytes()

                # Emit the frame to the client
                emit('video_feed', frame, broadcast=True, binary=True)
                socketio.sleep(0.003)

    except Exception as e:
        text_to_speech(f"Error occurred: {str(e)}")
        emit('video_feed', b'', broadcast=True)
    finally:
        cv2.destroyAllWindows()

@app.route('/subtraction')
def subtraction():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('subtraction.html')


@socketio.on('subtraction')
def subtraction():
    try:
        mp_hands = mp.solutions.hands
        cap = cv2.VideoCapture(0)
        global is_connected
        is_connected = True
        if not cap.isOpened():
            text_to_speech("cap is not opened")
            emit('video_feed', b'', broadcast=True)  # Notify client of failure
            return

        with mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5, ) as hands:

            while cap.isOpened():
                ret, image = cap.read()
                if not ret:
                    text_to_speech("no ret")
                    break

                if check_Thumb(image):
                    cap.release()
                    text_to_speech("Thumbs up detected! Redirecting.")
                    emit('redirect', {"url": url_for('home')}, broadcast=True)
                    break

                if is_connected == False:
                    break

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                fingerCount1 = 0
                fingerCount2 = 0

                if results.multi_hand_landmarks:

                    for hand_landmarks in results.multi_hand_landmarks:
                        # Get hand index to check label (left or right)
                        handIndex = results.multi_hand_landmarks.index(hand_landmarks)
                        handLabel = results.multi_handedness[handIndex].classification[0].label

                        # Set variable to keep landmarks positions (x and y)
                        handLandmarks = []

                        # Fill list with x and y positions of each landmark
                        for landmarks in hand_landmarks.landmark:
                            handLandmarks.append([landmarks.x, landmarks.y])

                        if handLabel == "Left":
                            if handLandmarks[4][0] > handLandmarks[3][0]:
                                fingerCount1 += 1
                            if handLandmarks[8][1] < handLandmarks[6][1]:  # Index finger
                                fingerCount1 += 1
                            if handLandmarks[12][1] < handLandmarks[10][1]:  # Middle finger
                                fingerCount1 += 1
                            if handLandmarks[16][1] < handLandmarks[14][1]:  # Ring finger
                                fingerCount1 += 1
                            if handLandmarks[20][1] < handLandmarks[18][1]:  # Pinky
                                fingerCount1 += 1

                        elif handLabel == "Right":
                            if handLandmarks[4][0] < handLandmarks[3][0]:
                                fingerCount2 += 1
                            if handLandmarks[8][1] < handLandmarks[6][1]:  # Index finger
                                fingerCount2 += 1
                            if handLandmarks[12][1] < handLandmarks[10][1]:  # Middle finger
                                fingerCount2 += 1
                            if handLandmarks[16][1] < handLandmarks[14][1]:  # Ring finger
                                fingerCount2 += 1
                            if handLandmarks[20][1] < handLandmarks[18][1]:  # Pinky
                                fingerCount2 += 1


                # Display finger count
                sum = fingerCount1 - fingerCount2
                disp = str(fingerCount1) + ' - ' + str(fingerCount2) + ' = ' + str(sum)
                cv2.putText(image, disp, (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)
                _, buffer = cv2.imencode('.png', image)
                frame = buffer.tobytes()

                # Emit the frame to the client
                emit('video_feed', frame, broadcast=True, binary=True)
                socketio.sleep(0.003)

    except Exception as e:
        text_to_speech(f"Error occurred: {str(e)}")
        emit('video_feed', b'', broadcast=True)
    finally:
        cv2.destroyAllWindows()


@app.route('/multiplication')
def multiplication():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('multiplication.html')


@socketio.on('multiplication')
def multiplication():
    try:
        mp_hands = mp.solutions.hands
        cap = cv2.VideoCapture(0)
        global is_connected
        is_connected=True
        if not cap.isOpened():
            text_to_speech("Camera is not opened")
            emit('video_feed', b'', broadcast=True)  # Notify client of failure
            return

        with mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5, ) as hands:

            while cap.isOpened():
                ret, image = cap.read()
                if not ret:
                    text_to_speech("Camera capture failed")
                    break

                if check_Thumb(image):
                    cap.release()
                    text_to_speech("Thumbs up detected! Redirecting.")
                    emit('redirect', {"url": url_for('home')}, broadcast=True)
                    break
                if is_connected == False:
                    break

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                fingerCount1 = 0
                fingerCount2 = 0

                if results.multi_hand_landmarks:

                    for hand_landmarks in results.multi_hand_landmarks:
                        handIndex = results.multi_hand_landmarks.index(hand_landmarks)
                        handLabel = results.multi_handedness[handIndex].classification[0].label
                        handLandmarks = [[landmark.x, landmark.y] for landmark in hand_landmarks.landmark]

                        if handLabel == "Left":
                            fingerCount1 = sum(
                                [
                                    handLandmarks[4][0] > handLandmarks[3][0],
                                    handLandmarks[8][1] < handLandmarks[6][1],
                                    handLandmarks[12][1] < handLandmarks[10][1],
                                    handLandmarks[16][1] < handLandmarks[14][1],
                                    handLandmarks[20][1] < handLandmarks[18][1],
                                ]
                            )

                        elif handLabel == "Right":
                            fingerCount2 = sum(
                                [
                                    handLandmarks[4][0] < handLandmarks[3][0],
                                    handLandmarks[8][1] < handLandmarks[6][1],
                                    handLandmarks[12][1] < handLandmarks[10][1],
                                    handLandmarks[16][1] < handLandmarks[14][1],
                                    handLandmarks[20][1] < handLandmarks[18][1],
                                ]
                            )

                # Perform multiplication
                product = fingerCount1 * fingerCount2
                disp = f"{fingerCount1} x {fingerCount2} = {product}"

                # Draw multiplication grid
                grid_start_x, grid_start_y = 50, 50
                grid_cell_size = 40
                counter = 1
                for row in range(fingerCount1):
                    for col in range(fingerCount2):
                        top_left = (grid_start_x + col * grid_cell_size, grid_start_y + row * grid_cell_size)
                        bottom_right = (top_left[0] + grid_cell_size, top_left[1] + grid_cell_size)
                        cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)
                        cell_center = (top_left[0] + grid_cell_size // 2, top_left[1] + grid_cell_size // 2)

                        # Add number to the center of the cell (slightly adjusting for better alignment)
                        cv2.putText(
                            image,
                            str(counter),
                            (cell_center[0] - 10, cell_center[1] + 10),  # Adjusting x and y for centering
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,  # Font scale increased for better visibility
                            (0, 255, 0),  # Green color for text
                            2,  # Thicker font for better visibility
                            cv2.LINE_AA
                        )
                        counter += 1

                # Display text on the video feed
                cv2.putText(image, disp, (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)

                _, buffer = cv2.imencode('.png', image)
                frame = buffer.tobytes()

                # Emit the frame to the client
                emit('video_feed', frame, broadcast=True, binary=True)
                socketio.sleep(0.003)

    except Exception as e:
        text_to_speech(f"Error occurred: {str(e)}")
        emit('video_feed', b'', broadcast=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()


@app.route('/count')
def count():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('count.html')

@socketio.on('start_finger_counting')
def start_fingercounting():
    try:
        mp_hands = mp.solutions.hands
        cap = cv2.VideoCapture(0)
        global is_connected
        is_connected = True
        if not cap.isOpened():
            text_to_speech("cap is not opened")
            emit('video_feed', b'', broadcast=True)  # Notify client of failure
            return

        with mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5, max_num_hands=10) as hands:

            while cap.isOpened():
                ret, image = cap.read()
                if not ret:
                    text_to_speech("no ret")
                    break

                if check_Thumb(image):
                    cap.release()
                    text_to_speech("Thumbs up detected! Redirecting.")
                    emit('redirect', {"url": url_for('home')}, broadcast=True)
                    break
                if is_connected == False:
                    break

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                fingerCount = 0

                if results.multi_hand_landmarks:

                    for hand_landmarks in results.multi_hand_landmarks:
                        # Get hand index to check label (left or right)
                        handIndex = results.multi_hand_landmarks.index(hand_landmarks)
                        handLabel = results.multi_handedness[handIndex].classification[0].label

                        # Set variable to keep landmarks positions (x and y)
                        handLandmarks = []

                        # Fill list with x and y positions of each landmark
                        for landmarks in hand_landmarks.landmark:
                            handLandmarks.append([landmarks.x, landmarks.y])

                        if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
                            fingerCount += 1
                        elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
                            fingerCount += 1

                        if handLandmarks[8][1] < handLandmarks[6][1]:  # Index finger
                            fingerCount += 1
                        if handLandmarks[12][1] < handLandmarks[10][1]:  # Middle finger
                            fingerCount += 1
                        if handLandmarks[16][1] < handLandmarks[14][1]:  # Ring finger
                            fingerCount += 1
                        if handLandmarks[20][1] < handLandmarks[18][1]:  # Pinky
                            fingerCount += 1
                        if handLandmarks[8][1] < handLandmarks[6][1] and handLandmarks[20][1] < handLandmarks[18][1] and \
                                handLandmarks[16][1] > handLandmarks[14][1] and handLandmarks[12][1] > handLandmarks[10][1]:
                            f = True

                # Display finger count
                cv2.putText(image, str(fingerCount), (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)

                _, buffer = cv2.imencode('.png', image)
                frame = buffer.tobytes()

                # Emit the frame to the client
                emit('video_feed', frame, broadcast=True, binary=True)
                socketio.sleep(0.003)

    except Exception as e:
        text_to_speech(f"Error occurred: {str(e)}")
        emit('video_feed', b'', broadcast=True)
    finally:
        cv2.destroyAllWindows()

@app.route('/vwb')
def vwb():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('vwb.html')



def virtualwb():
    genai.configure(api_key="xxxxxxxxxxxxxxxxxxxxx")  # Replace with your actual API key
    model = genai.GenerativeModel('gemini-1.5-flash')
    prev_pos = None
    canvas = None
    global is_connected
    is_connected=True

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)  # Set width
    cap.set(4, 720)   # Set height

    while is_connected:
        success, img = cap.read()
        if not success:
            break
        img = cv2.flip(img, 1)
        img = cv2.GaussianBlur(img, (5, 5), 0)

        if canvas is None:
            canvas = np.zeros_like(img)

        info = getHandInfo(img)

        if info[0] is not None:
            if info[0] == [0, 1, 0, 0, 1] or info[0] == [1, 1, 0, 0, 1]:
                emit("redirect", {"url": url_for('home')}, broadcast=True)
                break
            prev_pos, canvas = draw(info, prev_pos, canvas)
            if is_thumbs_up(info[0], info[1]):
                output_text = sendToAI(model, canvas, info[0])
                socketio.emit('ai_response', {'text': output_text})  # Send AI response

        combined_image = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)

        _, buffer = cv2.imencode('.jpg', combined_image)
        frame_data = base64.b64encode(buffer).decode('utf-8')
        socketio.emit('video_frame', {'frame': frame_data})  # Send video frame

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cap.release()


@socketio.on('start_virtual_whiteboard')
def start_virtual_whiteboard():
    virtualwb()





@app.route('/games')
def games():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('games.html')

@app.route('/tictactoe')
def tictactoe():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('tictactoe.html')

@app.route('/tictactoe/computer')
def tictactoe_computer():
    return render_template('tictactoe_computer.html')

@app.route('/tictactoe/friend')
def tictactoe_friend():
    return render_template('tictactoe_friend.html')

@app.route('/sitstand')
def sitstand_page():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('sitstand.html')


@app.route('/inout')
def inout_page():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('inout.html')


@socketio.on('sitstand_game')
def sitstand_game():
    global is_connected
    is_connected = True
    try:
        while is_connected:
            r = random.randint(1, 2)
            action = 'sit' if r == 1 else 'stand'
            text_to_speech(action)
            emit('game_action', action, broadcast=True)
            socketio.sleep(2)  # Delay between actions
    except Exception as e:
        text_to_speech(f"Error occurred: {str(e)}")


@socketio.on('inout_game')
def inout_game():
    global is_connected
    is_connected = True
    try:
        while is_connected:
            r = random.randint(1, 2)
            action = 'in' if r == 1 else 'out'
            text_to_speech(action)
            emit('game_action', action, broadcast=True)
            socketio.sleep(2)  # Delay between actions
    except Exception as e:
        text_to_speech(f"Error occurred: {str(e)}")

@app.route('/secretnumber')
def secret_number_page():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('secretnumber.html')


@socketio.on('start_secret_number_game')
def start_secret_number_game():
    secret_number = random.randint(1, 100)
    max_guesses = 10
    guesses_taken = 0

    def check_guess(guess):
        nonlocal guesses_taken
        nonlocal secret_number

        guesses_taken += 1
        if guesses_taken > max_guesses:
            return {"result": "Game Over", "message": f"The secret number was: {secret_number}"}

        try:
            guess = int(guess)
        except ValueError:
            return {"result": "Invalid", "message": "Please enter a valid number."}

        if guess < secret_number:
            return {"result": "Higher", "message": f"Guess higher! {max_guesses - guesses_taken} guesses left."}
        elif guess > secret_number:
            return {"result": "Lower", "message": f"Guess lower! {max_guesses - guesses_taken} guesses left."}
        else:
            return {"result": "Win", "message": f"Congratulations! You guessed it in {guesses_taken} attempts."}


    @socketio.on('make_guess')
    def handle_guess(data):
        guess = data.get("guess", "")
        response = check_guess(guess)
        emit('game_response', response)


@app.route('/guesscolor')
def guess_color_page():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('guesscolor.html')


@socketio.on('start_color_game')
def start_color_game():
    global score, timeleft, game_active
    if not game_active:  # Prevent duplicate game sessions
        game_active = True
        score = 0
        timeleft = 30
        random.shuffle(colours)
        next_word = colours[0]
        next_color = colours[1]
        emit('update_game', {
            'score': score,
            'next_word': next_word,
            'next_color': next_color,
            'timeleft': timeleft
        })


@socketio.on('check_color')
def check_color(data):
    global score, timeleft, colours, game_active
    if game_active:
        user_input = data.get('color', '').strip().lower()
        correct_color = colours[1].lower()

        if user_input == correct_color:
            score += 1

        if timeleft > 0:
            random.shuffle(colours)
            next_word = colours[0]
            next_color = colours[1]
            emit('update_game', {
                'score': score,
                'next_word': next_word,
                'next_color': next_color,
                'timeleft': timeleft
            })
        else:
            end_game()


@socketio.on('start_timer')
def start_timer():
    global timeleft, game_active
    if game_active:  # Ensure timer only starts once
        while timeleft > 0:
            time.sleep(1)
            timeleft -= 1
            emit('update_timer', {'timeleft': timeleft}, broadcast=True)
        end_game()


def end_game():
    global game_active
    game_active = False
    emit('game_over', {
        'score': score,
        'message': f"Game Over! Your final score is {score}"
    }, broadcast=True)

@app.route('/whackamole')
def whackamole():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('whackamole.html')

@app.route('/spotthedifference')
def spotthedifference():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('spotthedifference.html')


@app.route('/spot-the-difference-data')
def get_spot_the_difference_data():
    with open('static/data.json', 'r') as file:
        data = json.load(file)
    random_pair = random.choice(data['pairs'])
    return jsonify(random_pair)



@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.secret_key = secrets.token_hex(16)
    socketio.run(app, debug=True)




