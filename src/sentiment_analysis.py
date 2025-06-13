import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from textblob import TextBlob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns

class SentimentAnalyzer:
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize BERT model for financial sentiment
        print("Loading BERT model for financial sentiment analysis...")
        self.bert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.bert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.bert_pipeline = pipeline(
            "sentiment-analysis",
            model=self.bert_model,
            tokenizer=self.bert_tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
    def vader_sentiment(self, text):
        """Analyze sentiment using VADER"""
        scores = self.vader_analyzer.polarity_scores(text)
        
        # Determine overall sentiment
        if scores['compound'] >= 0.05:
            sentiment = 'positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
            
        return {
            'sentiment': sentiment,
            'compound': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu']
        }
    
    def textblob_sentiment(self, text):
        """Analyze sentiment using TextBlob"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
            
        return {
            'sentiment': sentiment,
            'polarity': polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def bert_sentiment(self, text):
        """Analyze sentiment using FinBERT"""
        try:
            # Truncate text if too long
            if len(text) > 512:
                text = text[:512]
                
            result = self.bert_pipeline(text)[0]
            
            return {
                'sentiment': result['label'].lower(),
                'confidence': result['score']
            }
        except Exception as e:
            print(f"Error in BERT sentiment analysis: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.0}
    
    def analyze_dataframe(self, df, text_column='processed_text'):
        """Perform sentiment analysis on entire dataframe"""
        print("Starting sentiment analysis...")
        
        # VADER sentiment analysis
        vader_results = df[text_column].apply(self.vader_sentiment)
        df['vader_sentiment'] = [result['sentiment'] for result in vader_results]
        df['vader_compound'] = [result['compound'] for result in vader_results]
        df['vader_positive'] = [result['positive'] for result in vader_results]
        df['vader_negative'] = [result['negative'] for result in vader_results]
        df['vader_neutral'] = [result['neutral'] for result in vader_results]
        
        # TextBlob sentiment analysis
        textblob_results = df[text_column].apply(self.textblob_sentiment)
        df['textblob_sentiment'] = [result['sentiment'] for result in textblob_results]
        df['textblob_polarity'] = [result['polarity'] for result in textblob_results]
        df['textblob_subjectivity'] = [result['subjectivity'] for result in textblob_results]
        
        # BERT sentiment analysis
        bert_results = df[text_column].apply(self.bert_sentiment)
        df['bert_sentiment'] = [result['sentiment'] for result in bert_results]
        df['bert_confidence'] = [result['confidence'] for result in bert_results]
        
        print("Sentiment analysis completed!")
        return df
    
    def create_sentiment_ensemble(self, df):
        """Create ensemble sentiment prediction"""
        sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
        
        # Convert sentiments to numeric
        df['vader_numeric'] = df['vader_sentiment'].map(sentiment_mapping)
        df['textblob_numeric'] = df['textblob_sentiment'].map(sentiment_mapping)
        df['bert_numeric'] = df['bert_sentiment'].map(sentiment_mapping)
        
        # Calculate weighted average (giving more weight to BERT)
        df['ensemble_score'] = (
            0.3 * df['vader_numeric'] + 
            0.2 * df['textblob_numeric'] + 
            0.5 * df['bert_numeric']
        )
        
        # Convert back to categorical
        df['ensemble_sentiment'] = df['ensemble_score'].apply(
            lambda x: 'positive' if x > 0.2 else ('negative' if x < -0.2 else 'neutral')
        )
        
        return df
    
    def visualize_sentiment_distribution(self, df):
        """Create sentiment distribution visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # VADER sentiment distribution
        df['vader_sentiment'].value_counts().plot(kind='bar', ax=axes[0,0], title='VADER Sentiment Distribution')
        axes[0,0].set_ylabel('Count')
        
        # TextBlob sentiment distribution
        df['textblob_sentiment'].value_counts().plot(kind='bar', ax=axes[0,1], title='TextBlob Sentiment Distribution')
        axes[0,1].set_ylabel('Count')
        
        # BERT sentiment distribution
        df['bert_sentiment'].value_counts().plot(kind='bar', ax=axes[1,0], title='FinBERT Sentiment Distribution')
        axes[1,0].set_ylabel('Count')
        
        # Ensemble sentiment distribution
        df['ensemble_sentiment'].value_counts().plot(kind='bar', ax=axes[1,1], title='Ensemble Sentiment Distribution')
        axes[1,1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('sentiment_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Sentiment distribution plot saved as 'sentiment_distribution.png'")
        
        return fig

if __name__ == "__main__":
    # Test sentiment analysis
    from data_preprocessing import TextPreprocessor, load_sample_data
    
    preprocessor = TextPreprocessor()
    df = load_sample_data()
    processed_df = preprocessor.preprocess_dataframe(df, 'headline')
    
    analyzer = SentimentAnalyzer()
    sentiment_df = analyzer.analyze_dataframe(processed_df)
    ensemble_df = analyzer.create_sentiment_ensemble(sentiment_df)
    
    print("\nSentiment Analysis Results:")
    print(ensemble_df[['headline', 'vader_sentiment', 'textblob_sentiment', 'bert_sentiment', 'ensemble_sentiment']].head())
    
    analyzer.visualize_sentiment_distribution(ensemble_df)
