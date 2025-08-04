"""
Streamlit front‑end for the DiseasePredictor project.

This module provides a simple web interface that lets users enter a list of
symptoms and predicts a likely disease based on the existing training data.
The underlying model is trained on the public `dataset.csv` included in the
repository.  Training happens lazily the first time the app is run to keep
start‑up times reasonable.

To launch the application locally, install the dependencies listed in
``requirements.txt`` and run::

    streamlit run app.py

This will start a local server at ``http://localhost:8501`` where you can
enter symptoms separated by commas (for example, ``fever, chills, cough``) and
receive a predicted disease.  The model uses a bag‑of‑words representation of
the symptom list and a random forest classifier from scikit‑learn to predict
the disease label.
"""

from __future__ import annotations

try:
    import streamlit as st  # type: ignore[import]
except ImportError:
    # Provide a minimal stub for environments where Streamlit isn't installed.
    # This allows importing the module to access the prediction logic without
    # requiring the UI dependencies.  Attempting to run ``main()`` without
    # Streamlit installed will raise a more helpful error.
    class _StreamlitStub:  # pragma: no cover - stub implementation
        def __getattr__(self, name):
            raise ImportError(
                "streamlit is required to run the web interface. Install it by running `pip install streamlit`."
            )

    st = _StreamlitStub()  # type: ignore
import pandas as pd

from functools import lru_cache
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


@lru_cache(maxsize=1)
def load_data() -> tuple[list[str], pd.Series]:
    """Load the symptom/disease dataset from GitHub.

    The dataset contains one row per training example.  Each row lists a
    disease name and up to 17 associated symptoms.  Missing symptom columns
    are left blank.  This function flattens each row into a single string of
    space‑separated symptom tokens and returns a list of these strings along
    with the corresponding disease labels.

    Returns
    -------
    tuple[list[str], pd.Series]
        A tuple containing the training texts and the target labels.
    """
    url = (
        "https://raw.githubusercontent.com/"
        "keshavgoel787/DiseasePredictor/main/dataset.csv"
    )
    data = pd.read_csv(url)
    # Replace NaN with empty strings for concatenation
    data = data.fillna("")
    # Combine all symptom columns into a space‑separated string
    symptom_cols = [
        col for col in data.columns if col.lower().startswith("symptom")
    ]
    texts: list[str] = []
    for _, row in data.iterrows():
        symptoms = [str(row[col]).strip() for col in symptom_cols if row[col]]
        texts.append(" ".join(symptoms))
    labels = data["Disease"]
    return texts, labels


@lru_cache(maxsize=1)
def train_model() -> tuple[CountVectorizer, RandomForestClassifier]:
    """Train a random forest classifier on the dataset.

    The dataset is loaded via :func:`load_data`.  A bag‑of‑words
    (CountVectorizer) representation is used to convert symptom strings into
    feature vectors.  A random forest classifier is trained on these
    vectors.

    Returns
    -------
    tuple[CountVectorizer, RandomForestClassifier]
        The fitted vectorizer and classifier.
    """
    texts, labels = load_data()
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X, labels)
    return vectorizer, clf


def predict_disease(symptoms: str) -> str:
    """Predict a disease from a comma‑separated symptom string.

    Parameters
    ----------
    symptoms : str
        A string of comma‑separated symptom names entered by the user.

    Returns
    -------
    str
        The predicted disease label.
    """
    vectorizer, clf = train_model()
    # Normalize and join symptoms into a single space‑separated string
    tokens = [s.strip().replace(",", " ") for s in symptoms.split(",") if s.strip()]
    text = " ".join(tokens)
    # Transform and predict
    X_input = vectorizer.transform([text])
    prediction = clf.predict(X_input)[0]
    return prediction



def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(page_title="Disease Predictor", page_icon="🩺")
    st.title("Disease Predictor")
    st.write(
        """
        Enter one or more symptoms separated by commas.  The application will
        predict a likely disease based on the training data.
        """
    )
    default_example = "itching, skin_rash, nodal_skin_eruptions"
    user_input = st.text_input(
        "Symptoms (comma‑separated)", value=default_example, help="E.g. fever, cough, nausea"
    )
    if st.button("Predict"):
        if not user_input.strip():
            st.warning("Please enter at least one symptom.")
        else:
            result = predict_disease(user_input)
            st.success(f"Predicted Disease: {result}")


if __name__ == "__main__":
    main()
