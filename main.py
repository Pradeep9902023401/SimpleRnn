import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="🎬 IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Custom CSS --------------------
st.markdown("""
<style>
.main-title{
    font-size:42px;
    font-weight:bold;
    color:#FF4B4B;
}
.sub-title{
    font-size:18px;
    color:#B0B0B0;
}
.result-box{
    padding:20px;
    border-radius:12px;
    background-color:#262730;
    margin-top:20px;
}
.footer{
    text-align:center;
    color:gray;
    font-size:14px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- Load Model --------------------
@st.cache_resource
def load_rnn_model():
    return load_model("simple_rnn_imdb.h5")

model = load_rnn_model()

# -------------------- Load IMDB Word Index --------------------
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# -------------------- Helper Functions --------------------
def decode_review(encoded_review):
    return ' '.join(
        [reverse_word_index.get(i - 3, '?') for i in encoded_review]
    )

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=500
    )
    return padded_review

# -------------------- Sidebar --------------------
with st.sidebar:

    st.title("🎬 IMDB Sentiment Analyzer")

    st.markdown("---")

    st.success("### Model Details")

    st.write("**Algorithm:** Simple RNN")
    st.write("**Dataset:** IMDB Movie Reviews")
    st.write("**Vocabulary:** 10,000 Words")
    st.write("**Input Length:** 500 Words")
    st.write("**Framework:** TensorFlow/Keras")

    st.markdown("---")

    st.info("""
### Prediction Labels

😊 Positive Review

😞 Negative Review
""")

    st.markdown("---")

    st.write("Developed using Streamlit")

# -------------------- Main Title --------------------
st.markdown(
    '<p class="main-title">🎬 IMDB Movie Review Sentiment Analysis</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">Predict whether a movie review is Positive or Negative using a trained Simple RNN model.</p>',
    unsafe_allow_html=True
)

st.divider()

# -------------------- Sample Reviews --------------------
with st.expander("📋 Sample Reviews"):

    col1, col2 = st.columns(2)

    with col1:

        if st.button("😊 Load Positive Review"):
            st.session_state["sample"] = (
                "This movie was absolutely fantastic. "
                "The acting was brilliant and the story was amazing."
            )

    with col2:

        if st.button("😞 Load Negative Review"):
            st.session_state["sample"] = (
                "Worst movie ever. "
                "It was boring, slow and a complete waste of time."
            )

default_text = st.session_state.get("sample", "")

# -------------------- User Input --------------------
user_input = st.text_area(
    "✍ Enter Movie Review",
    value=default_text,
    height=220,
    placeholder="Type your movie review here..."
)

# -------------------- Statistics --------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Characters", len(user_input))

with col2:
    st.metric("Words", len(user_input.split()))

with col3:
    st.metric("Maximum Length", "500")

st.divider()

# -------------------- Prediction --------------------
if st.button("🔍 Analyze Sentiment", use_container_width=True):

    if user_input.strip() == "":
        st.warning("⚠ Please enter a movie review.")
        st.stop()

    with st.spinner("Analyzing Review..."):

        processed_review = preprocess_text(user_input)

        prediction = model.predict(processed_review, verbose=0)

        score = float(prediction[0][0])

        if score >= 0.5:
            sentiment = "😊 Positive"
            confidence = score
        else:
            sentiment = "😞 Negative"
            confidence = 1 - score

    st.divider()

    st.subheader("Prediction Result")

    if sentiment == "😊 Positive":
        st.success(f"Sentiment : {sentiment}")
    else:
        st.error(f"Sentiment : {sentiment}")

    st.write("### Confidence")

    st.progress(int(confidence * 100))

    st.metric(
        "Confidence Score",
        f"{confidence*100:.2f}%"
    )

    st.metric(
        "Raw Model Output",
        f"{score:.4f}"
    )

    st.write("### Review Summary")

    st.info(f"""
**Characters:** {len(user_input)}

**Words:** {len(user_input.split())}

**Prediction Threshold:** 0.50
""")

st.divider()

st.markdown(
    '<p class="footer">❤️ Built with Streamlit | TensorFlow | Simple RNN | IMDB Dataset</p>',
    unsafe_allow_html=True
)