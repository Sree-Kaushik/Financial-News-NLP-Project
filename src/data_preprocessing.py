import pandas as pd
import numpy as np
import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

class TextPreprocessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Add financial-specific stop words
        financial_stops = {'company', 'stock', 'market', 'financial', 'business', 
                          'report', 'quarter', 'year', 'million', 'billion'}
        self.stop_words.update(financial_stops)
    
    def clean_text(self, text):
        """Basic text cleaning"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text using NLTK"""
        tokens = word_tokenize(text)
        
        # Remove stop words and lemmatize
        lemmatized_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return lemmatized_tokens
    
    def spacy_preprocessing(self, text):
        """Advanced preprocessing using spaCy"""
        doc = self.nlp(text)
        
        # Extract tokens that are not stop words, punctuation, or spaces
        processed_tokens = [
            token.lemma_.lower() 
            for token in doc 
            if not token.is_stop 
            and not token.is_punct 
            and not token.is_space 
            and len(token.text) > 2
            and token.is_alpha
        ]
        
        return processed_tokens
    
    def extract_entities(self, text):
        """Extract named entities using spaCy"""
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities
    
    def preprocess_dataframe(self, df, text_column):
        """Preprocess entire dataframe"""
        print("Starting text preprocessing...")
        
        # Clean text
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # Tokenize and lemmatize
        df['tokens_nltk'] = df['cleaned_text'].apply(self.tokenize_and_lemmatize)
        df['tokens_spacy'] = df['cleaned_text'].apply(self.spacy_preprocessing)
        
        # Extract entities
        df['entities'] = df['cleaned_text'].apply(self.extract_entities)
        
        # Create processed text for analysis
        df['processed_text'] = df['tokens_spacy'].apply(lambda x: ' '.join(x))
        
        print("Text preprocessing completed!")
        return df

def load_sample_data():
    """Create sample financial news data"""
    sample_data = [
        "Apple Inc. reported strong quarterly earnings, beating analyst expectations with revenue growth of 15%.",
        "Tesla stock plummeted after disappointing delivery numbers were announced by the company.",
        "The Federal Reserve announced interest rate cuts to stimulate economic growth amid recession fears.",
        "Goldman Sachs upgraded its rating on Microsoft following strong cloud computing performance.",
        "Oil prices surged due to geopolitical tensions in the Middle East affecting global supply chains.",
        "Amazon's quarterly report showed declining profits despite increased revenue from e-commerce operations.",
        "The cryptocurrency market experienced significant volatility with Bitcoin dropping below $30,000.",
        "JPMorgan Chase reported record profits in the banking sector driven by higher interest rates.",
        "Google's parent company Alphabet faced regulatory scrutiny over antitrust concerns in Europe.",
        "The housing market showed signs of cooling with mortgage rates reaching multi-year highs."
    ]
    
    df = pd.DataFrame({
        'headline': sample_data,
        'date': pd.date_range('2024-01-01', periods=len(sample_data), freq='D'),
        'source': ['Financial Times', 'Reuters', 'Bloomberg', 'WSJ', 'CNBC'] * 2
    })
    
    return df

if __name__ == "__main__":
    # Test the preprocessing
    preprocessor = TextPreprocessor()
    df = load_sample_data()
    processed_df = preprocessor.preprocess_dataframe(df, 'headline')
    print(processed_df.head())
