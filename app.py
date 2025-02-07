import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, Dense

@st.cache_resource
def load_model_and_tokenizer():
    """Load model and tokenizer with compatibility handling"""
    try:
        # Custom model loading to handle older format
        with tf.keras.utils.custom_object_scope({'InputLayer': tf.keras.layers.InputLayer}):
            try:
                # First attempt: Try loading with custom objects
                model = tf.keras.models.load_model('WordPred.h5', compile=False)
            except:
                # Second attempt: Rebuild model architecture
                model = Sequential([
                    InputLayer(input_shape=(14,1)),  # Replace batch_shape with input_shape
                    LSTM(128, return_sequences=True),
                    LSTM(128),
                    Dense(1000, activation='softmax')  # Adjust units based on your vocabulary size
                ])
                # Load weights only
                model.load_weights('WordPred.h5')
        
        # Recompile the model
        model.compile(loss='categorical_crossentropy', 
                     optimizer='adam', 
                     metrics=['accuracy'])
        
        # Load tokenizer
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
            
        return model, tokenizer
    
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {str(e)}")
        return None, None

def predict_next_word(model, tokenizer, text, max_sequence_length):
    """Predict the next word given input text"""
    try:
        # Convert text to sequence
        token_list = tokenizer.texts_to_sequences([text])[0]
        
        # Handle sequence length
        if len(token_list) >= max_sequence_length:
            token_list = token_list[-(max_sequence_length-1):]
        
        # Pad sequence
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        
        # Make prediction
        prediction = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(prediction[0])
        
        # Find the word
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                return word
        
        return None
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

def main():
    st.title("Next Word Prediction with LSTM")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    if model is None or tokenizer is None:
        st.error("Failed to load model or tokenizer. Please check your files.")
        return
    
    # Get input
    input_text = st.text_input("Enter the sequence of words:", "To be or not to be")
    
    if st.button("Predict Next Word"):
        if not input_text.strip():
            st.warning("Please enter some text.")
            return
            
        try:
            # Get max sequence length from model
            max_sequence_length = model.input_shape[1] + 1
            
            # Make prediction
            with st.spinner("Predicting..."):
                next_word = predict_next_word(model, tokenizer, input_text, max_sequence_length)
            
            if next_word:
                st.success(f"Next word prediction: {next_word}")
                # Show complete sequence
                st.write(f"Complete sequence: {input_text} **{next_word}**")
            else:
                st.warning("Couldn't predict the next word. Try a different input.")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Try using shorter input text or different words.")

if __name__ == "__main__":
    main()