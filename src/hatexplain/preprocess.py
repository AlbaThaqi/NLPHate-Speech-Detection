import re
import spacy
import nltk
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin

# Download stopwords the first time
nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

# Load spaCy English model once
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])


def clean_text(text: str) -> str:
    """
    Basic text normalization:
        - lowercase
        - remove URLs
        - remove mentions
        - remove hashtags
        - remove emojis
        - remove punctuation
    """
    text = text.lower()

    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # URLs
    text = re.sub(r"@\w+", "", text)                    # mentions
    text = re.sub(r"#\w+", "", text)                    # hashtags

    # Remove emojis and non-ASCII characters
    text = text.encode("ascii", "ignore").decode()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def lemmatize(text: str) -> str:
    """
    Lemmatize using spaCy.
    Removes stopwords and punctuation tokens.
    """
    doc = nlp(text)
    lemmas = [
        token.lemma_ for token in doc
        if token.lemma_ not in STOPWORDS and token.lemma_.isalpha()
    ]
    return " ".join(lemmas)


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer.
    Enables integration with Pipeline() and GridSearchCV.
    """

    def __init__(self, do_lemmatize=True):
        self.do_lemmatize = do_lemmatize

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cleaned = X.apply(clean_text)
        if self.do_lemmatize:
            cleaned = cleaned.apply(lemmatize)
        return cleaned
