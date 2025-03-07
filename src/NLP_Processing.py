import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy


class NLP_Preprocessing:

    def __init__(self, text):
        self.text = text
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

        # Initialize necessary objects
        self.stop_words = set(stopwords.words('english'))
        self.ps = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        os.system("python -m spacy download en_core_web_sm")
        self.model = spacy.load("en_core_web_sm")

    def preprocess_text(self):
        if self.text is None or self.text == '':
            return ''
        # 1. Tokenization
        tokens = word_tokenize(self.text)
        
        # 2. Lowercasing
        tokens = [token.lower() for token in tokens]
        
        # 3. Stopword Removal
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # 4. Normalization (Stemming and Lemmatization)
        self.tokens_stemmed = [self.ps.stem(token) for token in tokens]
        
        # Using lemmatization (example)
        self.tokens_lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]

        # 5. Handling Special Characters (Removing punctuation, numbers, etc.)
        self.tokens_cleaned = [token for token in self.tokens_lemmatized if token.isalpha()]

        # Return preprocessed text
        return " ".join(self.tokens_cleaned)
