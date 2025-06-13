import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go

class TextClusterer:
    def __init__(self):
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.vectorizer = None
        self.kmeans = None
        self.embeddings = None
        self.tsne_results = None
        self.pca_results = None
        
    def create_embeddings(self, texts):
        """Create sentence embeddings using SentenceTransformers"""
        print("Creating sentence embeddings...")
        self.embeddings = self.sentence_transformer.encode(texts)
        print(f"Embeddings shape: {self.embeddings.shape}")
        return self.embeddings
    
    def create_tfidf_features(self, texts):
        """Create TF-IDF features"""
        print("Creating TF-IDF features...")
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.8,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        return tfidf_matrix.toarray()
    
    def find_optimal_clusters(self, features, max_clusters=10):
        """Find optimal number of clusters using elbow method and silhouette score"""
        n_samples = features.shape[0]
        max_clusters = min(max_clusters, n_samples - 1)  # Ensure max_clusters is valid
        
        if max_clusters < 2:
            print(f"Not enough samples ({n_samples}) for clustering analysis")
            return 2
        
        inertias = []
        silhouette_scores = []
        cluster_range = range(2, max_clusters + 1)
        
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(features, cluster_labels))
        
        # Plot elbow curve and silhouette scores
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Elbow curve
        ax1.plot(cluster_range, inertias, marker='o')
        ax1.set_title('Elbow Method for Optimal Clusters')
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Inertia')
        ax1.grid(True)
        
        # Silhouette scores
        ax2.plot(cluster_range, silhouette_scores, marker='o', color='orange')
        ax2.set_title('Silhouette Score vs Number of Clusters')
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('cluster_optimization.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Cluster optimization plot saved as 'cluster_optimization.png'")
        
        # Find optimal clusters (highest silhouette score)
        optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters: {optimal_clusters}")
        print(f"Best silhouette score: {max(silhouette_scores):.3f}")
        
        return optimal_clusters
    
    def perform_clustering(self, features, n_clusters=5):
        """Perform K-means clustering"""
        print(f"Performing K-means clustering with {n_clusters} clusters...")
        
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(features)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(features, cluster_labels)
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        
        return cluster_labels
    
    def perform_tsne(self, features, perplexity=30):
        """Perform t-SNE dimensionality reduction"""
        print("Performing t-SNE dimensionality reduction...")
        
        # Adjust perplexity based on sample size
        n_samples = features.shape[0]
        # t-SNE requires perplexity < n_samples
        max_perplexity = max(1, n_samples - 1)
        adjusted_perplexity = min(perplexity, max_perplexity)
        
        # Ensure minimum perplexity of 1
        if adjusted_perplexity < 1:
            adjusted_perplexity = 1
        
        print(f"Using perplexity: {adjusted_perplexity} (adjusted from {perplexity} for {n_samples} samples)")
        
        tsne = TSNE(
            n_components=2,
            perplexity=adjusted_perplexity,
            random_state=42,
            n_iter=1000
        )
        
        self.tsne_results = tsne.fit_transform(features)
        print("t-SNE completed!")
        
        return self.tsne_results
    
    def perform_pca(self, features, n_components=2):
        """Perform PCA dimensionality reduction"""
        print("Performing PCA dimensionality reduction...")
        
        pca = PCA(n_components=n_components, random_state=42)
        self.pca_results = pca.fit_transform(features)
        
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        
        return self.pca_results
    
    def visualize_clusters_2d(self, df, method='tsne'):
        """Visualize clusters in 2D space"""
        if method == 'tsne' and self.tsne_results is not None:
            x, y = self.tsne_results[:, 0], self.tsne_results[:, 1]
            title = 't-SNE Visualization of Text Clusters'
        elif method == 'pca' and self.pca_results is not None:
            x, y = self.pca_results[:, 0], self.pca_results[:, 1]
            title = 'PCA Visualization of Text Clusters'
        else:
            print(f"No {method} results available. Please run {method} first.")
            return
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(x, y, c=df['cluster'], cmap='viridis', alpha=0.7)
        plt.colorbar(scatter)
        plt.title(title)
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        
        # Add cluster centers if available
        if self.kmeans is not None and method == 'pca':
            if self.kmeans.cluster_centers_.shape[1] == len(self.pca_results[0]):
                centers = self.kmeans.cluster_centers_
                plt.scatter(centers[:, 0], centers[:, 1], 
                           c='red', marker='x', s=200, linewidths=3, label='Centroids')
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{method}_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{method.upper()} clusters plot saved as '{method}_clusters.png'")
    
    def create_interactive_plot(self, df, method='tsne'):
        """Create interactive plot using Plotly"""
        if method == 'tsne' and self.tsne_results is not None:
            x, y = self.tsne_results[:, 0], self.tsne_results[:, 1]
            title = 't-SNE Interactive Visualization'
        elif method == 'pca' and self.pca_results is not None:
            x, y = self.pca_results[:, 0], self.pca_results[:, 1]
            title = 'PCA Interactive Visualization'
        else:
            print(f"No {method} results available.")
            return
        
        # Create a new dataframe for plotting
        plot_df = pd.DataFrame({
            'x': x,
            'y': y,
            'cluster': df['cluster'].astype(str),
            'headline': df['headline'],
            'sentiment': df['vader_sentiment'],
            'topic': df['dominant_topic'] if 'dominant_topic' in df.columns else 'N/A'
        })
        
        # Create interactive scatter plot
        fig = px.scatter(
            plot_df,
            x='x', 
            y='y',
            color='cluster',
            hover_data=['headline', 'sentiment', 'topic'],
            title=title,
            labels={'x': f'{method.upper()} Component 1', 'y': f'{method.upper()} Component 2'}
        )
        
        fig.update_layout(
            width=800,
            height=600,
            showlegend=True
        )
        
        # Save as HTML instead of showing
        fig.write_html(f'{method}_interactive_plot.html')
        print(f"Interactive {method} plot saved as '{method}_interactive_plot.html'")
        return fig
    
    def analyze_clusters(self, df, text_column='headline'):
        """Analyze cluster characteristics"""
        print("\nCluster Analysis:")
        print("=" * 50)
        
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_data = df[df['cluster'] == cluster_id]
            
            print(f"\nCluster {cluster_id} ({len(cluster_data)} documents):")
            print("-" * 30)
            
            # Sample headlines
            print("Sample headlines:")
            for headline in cluster_data[text_column].head(3):
                print(f"  - {headline}")
            
            # Sentiment distribution
            sentiment_dist = cluster_data['vader_sentiment'].value_counts()
            print(f"\nSentiment distribution:")
            for sentiment, count in sentiment_dist.items():
                print(f"  {sentiment}: {count} ({count/len(cluster_data)*100:.1f}%)")
            
            # Topic distribution if available
            if 'dominant_topic' in cluster_data.columns:
                topic_dist = cluster_data['dominant_topic'].value_counts()
                print(f"\nTopic distribution:")
                for topic, count in topic_dist.head(3).items():
                    print(f"  Topic {topic}: {count} ({count/len(cluster_data)*100:.1f}%)")

if __name__ == "__main__":
    # Test clustering
    from data_preprocessing import TextPreprocessor, load_sample_data
    from sentiment_analysis import SentimentAnalyzer
    from topic_modeling import TopicModeler
    
    # Load and preprocess data
    preprocessor = TextPreprocessor()
    df = load_sample_data()
    processed_df = preprocessor.preprocess_dataframe(df, 'headline')
    
    # Add sentiment analysis
    analyzer = SentimentAnalyzer()
    sentiment_df = analyzer.analyze_dataframe(processed_df)
    
    # Add topic modeling
    topic_modeler = TopicModeler(num_topics=3)
    lda_model = topic_modeler.train_gensim_lda(processed_df['tokens_spacy'].tolist())
    topic_df = topic_modeler.assign_topics_to_documents(sentiment_df, processed_df['tokens_spacy'].tolist())
    
    # Clustering
    clusterer = TextClusterer()
    
    # Create embeddings
    embeddings = clusterer.create_embeddings(topic_df['processed_text'].tolist())
    
    # Find optimal clusters
    optimal_k = clusterer.find_optimal_clusters(embeddings, max_clusters=8)
    
    # Perform clustering
    cluster_labels = clusterer.perform_clustering(embeddings, n_clusters=optimal_k)
    topic_df['cluster'] = cluster_labels
    
    # Dimensionality reduction
    tsne_results = clusterer.perform_tsne(embeddings)
    pca_results = clusterer.perform_pca(embeddings)
    
    # Visualizations
    clusterer.visualize_clusters_2d(topic_df, method='tsne')
    clusterer.visualize_clusters_2d(topic_df, method='pca')
    
    # Interactive plot
    clusterer.create_interactive_plot(topic_df, method='tsne')
    
    # Cluster analysis
    clusterer.analyze_clusters(topic_df)
    
    print("\nClustering analysis completed!")
