import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Test basic functionality
print("Testing NLTK installation...")

# Test tokenization
text = "Apple Inc. reported strong quarterly earnings. The stock price increased significantly."
sentences = sent_tokenize(text)
words = word_tokenize(text)

print(f"Sentences: {len(sentences)}")
print(f"Words: {len(words)}")

# Test stop words
stop_words = set(stopwords.words('english'))
print(f"English stop words loaded: {len(stop_words)} words")

# Test VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()
sentiment = analyzer.polarity_scores("This is a great financial report!")
print(f"Sentiment analysis working: {sentiment}")

print("✓ NLTK installation verified successfully!")
