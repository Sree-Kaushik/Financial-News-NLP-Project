import pandas as pd
import numpy as np
from gensim import corpora, models
from gensim.models import LdaModel, CoherenceModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

class TopicModeler:
    def __init__(self, num_topics=5):
        self.num_topics = num_topics
        self.lda_model = None
        self.dictionary = None
        self.corpus = None
        self.sklearn_lda = None
        self.vectorizer = None
        
    def prepare_gensim_data(self, tokenized_texts):
        """Prepare data for Gensim LDA"""
        # Create dictionary and corpus
        self.dictionary = corpora.Dictionary(tokenized_texts)
        
        # Filter extremes
        self.dictionary.filter_extremes(no_below=1, no_above=0.9)
        
        # Create corpus
        self.corpus = [self.dictionary.doc2bow(text) for text in tokenized_texts]
        
        return self.dictionary, self.corpus
    
    def train_gensim_lda(self, tokenized_texts):
        """Train LDA model using Gensim"""
        print("Training Gensim LDA model...")
        
        # Prepare data
        self.dictionary, self.corpus = self.prepare_gensim_data(tokenized_texts)
        
        # Train LDA model
        self.lda_model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            random_state=42,
            passes=10,
            alpha='auto',
            per_word_topics=True,
            eval_every=1
        )
        
        print("Gensim LDA training completed!")
        return self.lda_model
    
    def train_sklearn_lda(self, processed_texts):
        """Train LDA model using scikit-learn"""
        print("Training scikit-learn LDA model...")
        
        # Vectorize the text
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=1,
            max_df=0.9,
            stop_words='english'
        )
        
        doc_term_matrix = self.vectorizer.fit_transform(processed_texts)
        
        # Train LDA model
        self.sklearn_lda = LatentDirichletAllocation(
            n_components=self.num_topics,
            random_state=42,
            max_iter=10,
            learning_method='online'
        )
        
        self.sklearn_lda.fit(doc_term_matrix)
        
        print("Scikit-learn LDA training completed!")
        return self.sklearn_lda
    
    def get_topic_words_gensim(self, num_words=10):
        """Get top words for each topic from Gensim model"""
        topics = []
        for topic_id in range(self.num_topics):
            topic_words = self.lda_model.show_topic(topic_id, topn=num_words)
            topics.append(topic_words)
        return topics
    
    def get_topic_words_sklearn(self, num_words=10):
        """Get top words for each topic from sklearn model"""
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(self.sklearn_lda.components_):
            top_words_idx = topic.argsort()[-num_words:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append(top_words)
            
        return topics
    
    def assign_topics_to_documents(self, df, tokenized_texts):
        """Assign dominant topic to each document"""
        print("Assigning topics to documents...")
        
        topic_assignments = []
        topic_probabilities = []
        
        for tokens in tokenized_texts:
            # Get topic distribution for document
            doc_bow = self.dictionary.doc2bow(tokens)
            doc_topics = self.lda_model.get_document_topics(doc_bow)
            
            if doc_topics:
                # Get dominant topic
                dominant_topic = max(doc_topics, key=lambda x: x[1])
                topic_assignments.append(dominant_topic[0])
                topic_probabilities.append(dominant_topic[1])
            else:
                topic_assignments.append(-1)
                topic_probabilities.append(0.0)
        
        df['dominant_topic'] = topic_assignments
        df['topic_probability'] = topic_probabilities
        
        return df
    
    def calculate_coherence_score(self, tokenized_texts):
        """Calculate coherence score for the model"""
        try:
            coherence_model = CoherenceModel(
                model=self.lda_model,
                texts=tokenized_texts,
                dictionary=self.dictionary,
                coherence='c_v'
            )
            
            coherence_score = coherence_model.get_coherence()
            print(f"Coherence Score: {coherence_score:.4f}")
            
            return coherence_score
        except Exception as e:
            print(f"Could not calculate coherence score: {e}")
            return 0.0
    
    def visualize_topics(self):
        """Create topic visualizations"""
        # Topic word clouds
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        topics = self.get_topic_words_gensim(20)
        
        for i, topic_words in enumerate(topics):
            if i < len(axes):
                # Create word cloud for topic
                word_freq = dict(topic_words)
                if word_freq:  # Check if word_freq is not empty
                    wordcloud = WordCloud(
                        width=400, 
                        height=300, 
                        background_color='white'
                    ).generate_from_frequencies(word_freq)
                    
                    axes[i].imshow(wordcloud, interpolation='bilinear')
                    axes[i].set_title(f'Topic {i+1}', fontsize=14, fontweight='bold')
                    axes[i].axis('off')
                else:
                    axes[i].text(0.5, 0.5, f'Topic {i+1}\n(No words)', 
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].axis('off')
        
        # Remove empty subplots
        for i in range(len(topics), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig('topic_wordclouds.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Topic wordclouds saved as 'topic_wordclouds.png'")
        
        return fig
    
    def create_topic_distribution_chart(self, df):
        """Create topic distribution chart"""
        topic_counts = df['dominant_topic'].value_counts().sort_index()
        
        plt.figure(figsize=(10, 6))
        topic_counts.plot(kind='bar')
        plt.title('Document Distribution Across Topics')
        plt.xlabel('Topic')
        plt.ylabel('Number of Documents')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig('topic_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Topic distribution chart saved as 'topic_distribution.png'")
    
    def find_optimal_topics(self, tokenized_texts, max_topics=10):
        """Find optimal number of topics using coherence score"""
        coherence_scores = []
        topic_ranges = range(2, max_topics + 1)
        
        for num_topics in topic_ranges:
            print(f"Testing {num_topics} topics...")
            
            # Temporary model
            temp_lda = LdaModel(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=num_topics,
                random_state=42,
                passes=10
            )
            
            # Calculate coherence
            try:
                coherence_model = CoherenceModel(
                    model=temp_lda,
                    texts=tokenized_texts,
                    dictionary=self.dictionary,
                    coherence='c_v'
                )
                coherence_scores.append(coherence_model.get_coherence())
            except:
                coherence_scores.append(0.0)
        
        # Plot coherence scores
        plt.figure(figsize=(10, 6))
        plt.plot(topic_ranges, coherence_scores, marker='o')
        plt.title('Coherence Score vs Number of Topics')
        plt.xlabel('Number of Topics')
        plt.ylabel('Coherence Score')
        plt.grid(True)
        plt.savefig('coherence_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Coherence scores plot saved as 'coherence_scores.png'")
        
        # Find optimal number of topics
        optimal_topics = topic_ranges[np.argmax(coherence_scores)]
        print(f"Optimal number of topics: {optimal_topics}")
        
        return optimal_topics, coherence_scores

if __name__ == "__main__":
    # Test topic modeling
    from data_preprocessing import TextPreprocessor, load_sample_data
    
    preprocessor = TextPreprocessor()
    df = load_sample_data()
    processed_df = preprocessor.preprocess_dataframe(df, 'headline')
    
    # Topic modeling
    topic_modeler = TopicModeler(num_topics=3)
    
    # Train models
    lda_model = topic_modeler.train_gensim_lda(processed_df['tokens_spacy'].tolist())
    sklearn_lda = topic_modeler.train_sklearn_lda(processed_df['processed_text'].tolist())
    
    # Assign topics
    topic_df = topic_modeler.assign_topics_to_documents(processed_df, processed_df['tokens_spacy'].tolist())
    
    # Calculate coherence
    coherence = topic_modeler.calculate_coherence_score(processed_df['tokens_spacy'].tolist())
    
    # Visualize
    topic_modeler.visualize_topics()
    topic_modeler.create_topic_distribution_chart(topic_df)
    
    print("\nTopic modeling completed!")
    print(f"Topics found: {topic_modeler.get_topic_words_gensim()}")
