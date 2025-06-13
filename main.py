import pandas as pd
import numpy as np
import os
import nltk
from src.data_preprocessing import TextPreprocessor, load_sample_data
from src.sentiment_analysis import SentimentAnalyzer
from src.topic_modeling import TopicModeler
from src.clustering import TextClusterer
from src.dashboard import FinancialNLPDashboard
import warnings
warnings.filterwarnings('ignore')

# Set environment variable to avoid tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Download NLTK data once at the start
def download_nltk_data():
    """Download required NLTK data packages once"""
    packages = ['punkt', 'stopwords', 'wordnet', 'vader_lexicon']
    for package in packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            nltk.download(package, quiet=True)

def main():
    """Main execution function"""
    print("=" * 60)
    print("FINANCIAL NEWS NLP ANALYSIS PIPELINE")
    print("=" * 60)
    
    # Download NLTK data once
    download_nltk_data()
    
    # Step 1: Data Loading and Preprocessing
    print("\n1. Loading and preprocessing data...")
    preprocessor = TextPreprocessor()
    df = load_sample_data()
    processed_df = preprocessor.preprocess_dataframe(df, 'headline')
    
    print(f"Loaded {len(processed_df)} articles")
    print("Sample processed text:")
    print(processed_df['processed_text'].iloc[0])
    
    # Step 2: Sentiment Analysis
    print("\n2. Performing sentiment analysis...")
    analyzer = SentimentAnalyzer()
    sentiment_df = analyzer.analyze_dataframe(processed_df)
    ensemble_df = analyzer.create_sentiment_ensemble(sentiment_df)
    
    # Display sentiment results
    print("\nSentiment Analysis Results:")
    sentiment_summary = ensemble_df['ensemble_sentiment'].value_counts()
    for sentiment, count in sentiment_summary.items():
        print(f"  {sentiment.capitalize()}: {count} ({count/len(ensemble_df)*100:.1f}%)")
    
    # Visualize sentiment distribution
    print("Generating sentiment visualizations...")
    analyzer.visualize_sentiment_distribution(ensemble_df)
    
    # Step 3: Topic Modeling
    print("\n3. Performing topic modeling...")
    topic_modeler = TopicModeler(num_topics=3)
    
    # Train LDA model
    lda_model = topic_modeler.train_gensim_lda(processed_df['tokens_spacy'].tolist())
    
    # Assign topics to documents
    topic_df = topic_modeler.assign_topics_to_documents(ensemble_df, processed_df['tokens_spacy'].tolist())
    
    # Calculate coherence score
    coherence = topic_modeler.calculate_coherence_score(processed_df['tokens_spacy'].tolist())
    
    # Display topic results
    print(f"\nTopic Modeling Results (Coherence Score: {coherence:.3f}):")
    topics = topic_modeler.get_topic_words_gensim(10)
    for i, topic_words in enumerate(topics):
        print(f"  Topic {i+1}: {', '.join([word for word, _ in topic_words[:5]])}")
    
    # Visualize topics
    topic_modeler.visualize_topics()
    topic_modeler.create_topic_distribution_chart(topic_df)
    
    # Step 4: Clustering and Dimensionality Reduction
    print("\n4. Performing clustering and dimensionality reduction...")
    clusterer = TextClusterer()
    
    # Create embeddings
    embeddings = clusterer.create_embeddings(topic_df['processed_text'].tolist())
    
    # Find optimal number of clusters
    optimal_k = clusterer.find_optimal_clusters(embeddings, max_clusters=6)
    
    # Perform clustering
    cluster_labels = clusterer.perform_clustering(embeddings, n_clusters=optimal_k)
    topic_df['cluster'] = cluster_labels
    
    # Dimensionality reduction
    tsne_results = clusterer.perform_tsne(embeddings)
    pca_results = clusterer.perform_pca(embeddings)
    
    # Add coordinates to dataframe for visualization
    topic_df['tsne_x'] = tsne_results[:, 0]
    topic_df['tsne_y'] = tsne_results[:, 1]
    topic_df['pca_x'] = pca_results[:, 0]
    topic_df['pca_y'] = pca_results[:, 1]
    
    # Visualize clusters
    clusterer.visualize_clusters_2d(topic_df, method='tsne')
    clusterer.visualize_clusters_2d(topic_df, method='pca')
    
    # Create interactive plot
    clusterer.create_interactive_plot(topic_df, method='tsne')
    
    # Analyze clusters
    clusterer.analyze_clusters(topic_df)
    
    # Step 5: Save Results
    print("\n5. Saving results...")
    topic_df.to_csv('financial_nlp_results.csv', index=False)
    print("Results saved to 'financial_nlp_results.csv'")
    
    # Step 6: Launch Dashboard
    print("\n6. Launching interactive dashboard...")
    print("Dashboard will open in your browser at http://localhost:8050")
    print("Press Ctrl+C to stop the dashboard server")
    
    dashboard = FinancialNLPDashboard(topic_df)
    dashboard.run_server(debug=False, port=8050)

if __name__ == "__main__":
    main()
