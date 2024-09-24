import joblib
import tensorflow as tf

# import model and word2idx
model_lstm = tf.keras.models.load_model('artifacts/model.h5')
word2idx = joblib.load('artifacts/word2idx.joblib')
idx2word = joblib.load('artifacts/idx2word.joblib')

def predict_new(text: str):
    """ function for inference new samples of text"""

    # Encode the text
    text_encoded = [word2idx.get(word, 0) for word in text.split()]

    # Padding and truncating
    text_padded = tf.keras.preprocessing.sequence.pad_sequences([text_encoded], 
                                                                maxlen=100, 
                                                                padding='post', 
                                                                truncating='post', 
                                                                value=0)
    
    # Model prediction
    prediction = ['positive' if model_lstm.predict(text_padded)[0][0] >= 0.5 else 'Negative'][0]

    return prediction