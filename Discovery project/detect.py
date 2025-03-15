import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2  # OpenCV for webcam streaming
import time  # For timing the frame capture
from groq import Groq  # Import Groq client
from openai import OpenAI  # Import OpenAI client
from dotenv import load_dotenv
import os
import shelve

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load environment variables for Groq and OpenAI APIs
load_dotenv()

# Load the animal detection model
animal_model = load_model("keras_Model.h5", compile=False)

# Load the labels for animal detection
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Initialize Groq and OpenAI clients
USER_AVATAR = "👤"
BOT_AVATAR = "🤖"
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to preprocess and predict the image for animal detection
def predict_image(image):
    # Resize and preprocess the image
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)  # Convert image to numpy array
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1  # Normalize

    # Load the image into the array
    data[0] = normalized_image_array

    # Predict the model
    prediction = animal_model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name[2:], confidence_score

# Inject custom CSS to set the background color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #E0F7FA;  /* Light Blue */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for navigation and chatbot selection
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a mode:", ("Animal Detection", "Chatbot"))
chatbot_type = st.sidebar.selectbox("Choose a chatbot:", ("RUDE Chatbot", "GPT Chatbot"))  # Updated options

# Animal Detection Section
if app_mode == "Animal Detection":
    st.markdown("<h1 style='text-align: center; color: #FF5733;'>Animal Detection using AI</h1>", unsafe_allow_html=True)

    # Option to choose between uploading an image or using the webcam
    option = st.radio("Choose an option:", ("Upload an image", "Use webcam"))

    if option == "Upload an image":
        # Upload image through Streamlit
        uploaded_file = st.file_uploader("Upload an image of an animal...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Open the uploaded image
            image = Image.open(uploaded_file).convert("RGB")

            # Display the uploaded image
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Predict the image
            class_name, confidence_score = predict_image(image)

            # Display the result
            st.write("### Prediction:")
            st.write(f"**Class:** {class_name}")
            st.write(f"**Confidence Score:** {confidence_score:.2f}")
    else:
        # Streaming webcam
        st.write("### Use your webcam for real-time animal detection")

        # Initialize OpenCV webcam
        cap = cv2.VideoCapture(0)  # 0 for default webcam

        # Placeholder for displaying the webcam feed and prediction
        frame_placeholder = st.empty()
        prediction_placeholder = st.empty()

        # Initialize timer
        last_capture_time = time.time()

        # Streamlit loop for real-time updates
        while cap.isOpened():
            # Capture frame-by-frame from the webcam
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from webcam.")
                break

            # Convert the frame from BGR (OpenCV) to RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            # Display the webcam feed
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            # Check if 10 seconds have passed since the last capture
            current_time = time.time()
            if current_time - last_capture_time >= 10:  # 10 seconds
                # Predict the animal in the frame
                class_name, confidence_score = predict_image(image)

                # Display the prediction
                prediction_placeholder.write(f"### Prediction: **{class_name}** (Confidence: {confidence_score:.2f})")

                # Update the last capture time
                last_capture_time = current_time

            # Break the loop if the user presses 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the webcam and close OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

# Chatbot Section
elif app_mode == "Chatbot":
    if chatbot_type == "RUDE Chatbot":  # Updated condition
        st.markdown("<h1 style='text-align: center; color: black; font-weight: bold;'>Welcome to RUDE AI</h1>", unsafe_allow_html=True)
        client = groq_client
        model_key = "groq_model"
        model_name = "mixtral-8x7b-32768"
    else:  # GPT Chatbot
        st.header("Welcome to GPT Chatbot")  # Updated header
        client = openai_client
        model_key = "openai_model"
        model_name = "gpt-3.5-turbo"

    # Ensure model is initialized in session state
    if model_key not in st.session_state:
        st.session_state[model_key] = model_name

    # Load chat history from shelve file
    def load_chat_history():
        with shelve.open("chat_history") as db:
            return db.get("messages", [])

    # Save chat history to shelve file
    def save_chat_history(messages):
        with shelve.open("chat_history") as db:
            db["messages"] = messages

    # Initialize or load chat history
    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_history()

    # Sidebar with a button to delete chat history
    with st.sidebar:
        if st.button("Delete Chat History"):
            st.session_state.messages = []
            save_chat_history([])

    # Display chat messages
    for message in st.session_state.messages:
        avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Main chat interface
    if prompt := st.chat_input("How can I help?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=BOT_AVATAR):
            message_placeholder = st.empty()
            full_response = ""
            for response in client.chat.completions.create(
                model=st.session_state[model_key],
                messages=st.session_state["messages"],
                stream=True,
            ):
                full_response += response.choices[0].delta.content or ""
                message_placeholder.markdown(full_response + "|")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Save chat history after each interaction
        save_chat_history(st.session_state.messages)