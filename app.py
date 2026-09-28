import streamlit as st
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout

st.set_page_config(
    page_title="Sleep Apnea Detection",
    page_icon="😴"
)

st.title("😴 Sleep Apnea Detection Using RNN")

st.write(
    "This application demonstrates an RNN model "
    "using synthetic breathing signals."
)

@st.cache_resource
def create_model():
    model = Sequential([
        SimpleRNN(
            32,
            activation="tanh",
            input_shape=(100, 1)
        ),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

model = create_model()

st.warning(
    "Demo only: This model is not trained for clinical "
    "sleep apnea detection."
)

if st.button("Generate Demo Prediction"):

    signal = (
        0.1 * np.sin(
            2 * np.pi * 0.25 * np.linspace(0, 10, 100)
        )
        + 0.1 * np.random.randn(100)
    )

    signal = signal.reshape(1, 100, 1)

    prediction = model.predict(signal, verbose=0)[0][0]

    st.write("Demo model output:", float(prediction))

    st.info(
        "This output is from an untrained model and "
        "must not be interpreted as a medical prediction."
    )
