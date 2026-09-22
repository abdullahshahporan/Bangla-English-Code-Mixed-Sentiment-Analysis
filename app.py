"""Interactive web interface for the NLP sentiment analysis system.

Start the interface from the project folder with:
    python -m streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.prediction import predict_sentiment
from src.preprocessing import preprocess_text


# The page configuration must be the first Streamlit command.
st.set_page_config(
    page_title="Bangla-English Sentiment Analyzer",
    page_icon="💬",
    layout="centered",
)


st.title("Bangla-English Sentiment Analyzer")
st.write(
    "Enter Bangla-English code-mixed text to see the NLP preprocessing steps "
    "and the final four-class sentiment prediction."
)

with st.sidebar:
    st.header("Project information")
    st.write("**Corpus:** BnSentMix")
    st.write("**Classes:** Positive, Negative, Neutral, Mixed")
    st.write("**NLP flow:** Cleaning → Tokenization → Representation → Classification")
    st.info(
        "The final answer combines word and character TF-IDF, Word2Vec, "
        "BiLSTM, and Transformer probabilities. A clearly stronger negative "
        "clause after 'kintu' can decide a Mixed result."
    )

user_text = st.text_area(
    "Input sentence",
    height=120,
    placeholder="Example: product ta good but delivery onek late",
)

if st.button("Analyze sentiment", type="primary", width="stretch"):
    if not user_text.strip():
        st.warning("Please enter a sentence before clicking Analyze sentiment.")
    else:
        # Show the language-processing result before the model prediction.
        processed_text = preprocess_text(user_text)
        tokens = processed_text.split()

        st.subheader("1. NLP preprocessing")
        st.write("**Original text**")
        st.write(user_text)
        st.write("**Processed text**")
        st.code(processed_text, language=None)
        st.write(f"**Tokens ({len(tokens)}):** {tokens}")

        try:
            with st.spinner("Analyzing the sentence..."):
                result = predict_sentiment(user_text)

            st.subheader("2. Sentiment result")
            st.success(f"Predicted sentiment: **{result['predicted_sentiment']}**")
            st.write(f"**Model:** {result['model']}")

            probability_table = pd.DataFrame(
                {
                    "Sentiment": list(result["probabilities"].keys()),
                    "Probability (%)": [
                        probability * 100
                        for probability in result["probabilities"].values()
                    ],
                }
            ).set_index("Sentiment")

            st.subheader("3. Class probabilities")
            st.bar_chart(probability_table, y="Probability (%)")
            st.dataframe(
                probability_table.style.format("{:.2f}%"),
                width="stretch",
            )
        except FileNotFoundError as error:
            st.error(
                "A required ensemble artifact was not found. Follow the training "
                "commands shown in README.md."
            )
            st.caption(str(error))
        except Exception as error:  # Keep the interactive application responsive.
            st.error("Prediction could not be completed. See the technical message below.")
            st.exception(error)
