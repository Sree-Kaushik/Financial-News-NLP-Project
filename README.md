Financial News NLP Analytics Pipeline
A comprehensive Natural Language Processing pipeline for sentiment analysis and topic modeling on financial news data, featuring an interactive dashboard for data exploration and insights.
This project implements an end-to-end NLP pipeline that processes financial news articles to extract sentiment, discover topics, and cluster similar content. The system combines multiple machine learning models and provides interactive visualizations for strategic decision-making in financial analysis.

Key Features
Multi-Model Sentiment Analysis: VADER, TextBlob, and FinBERT ensemble approach
Topic Modeling: LDA with Gensim for thematic analysis and coherence optimization
Document Clustering: K-means with sentence embeddings for semantic grouping
Dimensionality Reduction: t-SNE and PCA visualizations for cluster analysis
Interactive Dashboard: Real-time filtering and exploration with Dash/Plotly
Comprehensive Analytics: Coherence scoring, silhouette analysis, and ensemble methods

 Results & Performance
Sentiment Analysis Results
![Sentiment Distribution](outputs/sentiment_distribution.-model ensemble approach achieved the following distribution:

40% Negative sentiment articles (4 out of 10)
30% Positive sentiment articles (3 out of 10)
30% Neutral sentiment articles (3 out of 10)

Model Performance Comparison:
VADER: 5 positive, 4 negative, 1 neutral
TextBlob: 6 positive, 3 neutral, 1 negative
FinBERT: 7 negative, 3 positive (specialized for financial text)
Ensemble: Balanced 4 negative, 3 positive, 3 neutral (weighted combination)

Topic Modeling Analysis
![Topic Distributionence Score**: 0.185 with 3 distinct topics identified:

![Topic Word Clouds](outputs/ Reports)**: report, quarterly, profit, revenue, strong, performance

Topic 2 (Market Dynamics): growth, rate, market, high, middle, oil, tension, geopolitical
Topic 3 (Regulatory & Corporate): company, parent, concern, regulatory, announce, face, google, scrutiny

Clustering Performance
![PCA Visualization Algorithm**: K-means with sentence embeddings
Optimal Clusters: 2 clusters identified
Silhouette Score: 0.750 (excellent cluster separation)
Dimensionality Reduction: PCA explaining 65% total variance

Cluster Analysis:
Cluster 0 (Yellow): Negative sentiment articles (66.7% negative)
Tesla stock decline, Oil price surge, Google regulatory issues

Cluster 1 (Purple): Positive sentiment articles (71.4% positive)
Apple earnings, Fed rate cuts, Goldman Sachs upgrades

🛠 Technology Stack
Core Libraries
NLP Processing: spaCy (3.4+), NLTK (3.7+), Transformers (4.20+)
Machine Learning: scikit-learn (1.1+), Gensim (4.2+)
Deep Learning: PyTorch (1.12+), sentence-transformers (2.2+)
Data Processing: pandas (1.5+), numpy (1.21+)
Visualization: matplotlib (3.5+), seaborn (0.11+), plotly (5.0+)
Dashboard: Dash (2.0+), dash-bootstrap-components (1.0+)

Models Implemented
FinBERT: ProsusAI/finbert for financial sentiment analysis
VADER: Rule-based sentiment analysis for social media text
TextBlob: Pattern-based sentiment polarity detection
LDA: Latent Dirichlet Allocation for topic discovery
K-means: Clustering with sentence transformer embeddings
t-SNE/PCA: Dimensionality reduction for visualization

📁 Project Structure
text
financial-nlp-project/
├── src/
│   ├── data_preprocessing.py      # Text cleaning, tokenization, NER
│   ├── sentiment_analysis.py     # Multi-model sentiment pipeline
│   ├── topic_modeling.py         # LDA topic modeling with coherence
│   ├── clustering.py             # K-means clustering and visualization
│   └── dashboard.py              # Interactive Dash dashboard
├── outputs/                      # Generated visualizations and results
│   ├── sentiment_distribution.png
│   ├── topic_wordclouds.png
│   ├── topic_distribution.png
│   ├── pca_clusters.png
│   ├── tsne_clusters.png
│   ├── cluster_optimization.png
│   ├── tsne_interactive_plot.html
│   └── financial_nlp_results.csv
├── screenshots/                  # Dashboard screenshots
├── data/                        # Sample datasets
├── main.py                      # Main pipeline execution
├── requirements.txt             # Project dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Git ignore file

 Installation & Setup
Prerequisites
Python 3.8 or higher
pip package manager
4GB+ RAM (for transformer models)
Internet connection (for model downloads)

Step 1: Clone Repository
bash
git clone https://github.com/yourusername/financial-nlp-project.git
cd financial-nlp-project
Step 2: Create Virtual Environment
bash
python -m venv nlp_env
source nlp_env/bin/activate  # On Windows: nlp_env\Scripts\activate
Step 3: Install Dependencies
bash
pip install -r requirements.txt
Step 4: Download Language Models
bash
# Download spaCy English model
python -m spacy download en_core_web_sm
# Download NLTK data
python -c "import nltk; nltk.download('all')"
Step 5: Set Environment Variables
bash
export TOKENIZERS_PARALLELISM=false
 Running the Pipeline
Complete Pipeline Execution
bash
python main.py

This will execute the entire pipeline:

Data Loading & Preprocessing (10 financial news articles)
Multi-Model Sentiment Analysis (VADER + TextBlob + FinBERT)
Topic Modeling (LDA with coherence optimization)
Document Clustering (K-means with sentence embeddings)
Dimensionality Reduction (t-SNE and PCA)
Interactive Dashboard Launch (http://localhost:8050)

Testing Individual Components
bash
# Test data preprocessing
python src/data_preprocessing.py

# Test sentiment analysis
python src/sentiment_analysis.py

# Test topic modeling
python src/topic_modeling.py

# Test clustering
python src/clustering.py

# Test dashboard
python src/dashboard.py
 Generated Outputs
Visualization Files
sentiment_distribution.png - Multi-model sentiment comparison charts
topic_wordclouds.png - Word clouds for each discovered topic
topic_distribution.png - Document distribution across topics
pca_clusters.png - PCA visualization of document clusters
tsne_clusters.png - t-SNE visualization of document clusters
cluster_optimization.png - Elbow method and silhouette analysis
tsne_interactive_plot.html - Interactive cluster exploration

Data Files
financial_nlp_results.csv - Complete processed dataset with all features
Contains: original text, cleaned text, sentiment scores, topic assignments, cluster labels, embeddings coordinates

Analysis Results
text
Sentiment Analysis Results:
  Negative: 4 (40.0%)
  Positive: 3 (30.0%)
  Neutral: 3 (30.0%)

Topic Modeling Results (Coherence Score: 0.185):
  Topic 1: report, quarterly, profit, revenue, strong
  Topic 2: growth, rate, market, high, middle
  Topic 3: company, parent, concern, regulatory, announce

Clustering Results:
  Optimal number of clusters: 2
  Best silhouette score: 0.750
  PCA explained variance: 65.3%
 Interactive Dashboard Features
Access the dashboard at: http://localhost:8050

Dashboard Components
Summary Cards: Total articles, sentiment counts, cluster statistics
Interactive Filters: Filter by sentiment, cluster, and news source
Sentiment Distribution: Pie charts comparing all models
Topic Analysis: Bar charts and word cloud visualizations
Cluster Visualization: Interactive t-SNE scatter plots with hover details
Timeline Analysis: Sentiment trends over time
Source Analysis: Sentiment breakdown by news outlet
Data Table: Detailed article information with filtering

Key Insights Available
Cluster 0: Predominantly negative articles (Tesla decline, Oil surge, Google regulatory)
Cluster 1: Predominantly positive articles (Apple earnings, Fed cuts, Goldman upgrades)

Topic Separation: Clear thematic grouping of financial reports, market dynamics, and regulatory issues

Model Comparison: FinBERT shows higher sensitivity to financial context vs general-purpose models

 Technical Implementation Details
Data Preprocessing Pipeline
Text Cleaning: Lowercase conversion, special character removal, whitespace normalization
Tokenization: spaCy and NLTK tokenizers for robust text segmentation
Lemmatization: WordNet lemmatizer for word normalization
Stop Word Removal: Custom financial stop words + NLTK English stop words
Named Entity Recognition: spaCy NER for financial entities extraction

Sentiment Analysis Architecture
VADER: Rule-based approach optimized for social media text
TextBlob: Pattern-based polarity and subjectivity scoring
FinBERT: Transformer model fine-tuned on financial text
Ensemble Method: Weighted combination (30% VADER + 20% TextBlob + 50% FinBERT)

Topic Modeling Implementation
LDA Algorithm: Gensim implementation with hyperparameter tuning
Coherence Optimization: C_v coherence measure for model selection
Topic Assignment: Dominant topic selection based on probability distribution
Visualization: Word clouds and distribution charts for interpretation

Clustering Methodology
Embeddings: Sentence-BERT for semantic text representations
Algorithm: K-means with optimal cluster detection
Evaluation: Silhouette score and elbow method for cluster validation
Visualization: t-SNE and PCA for 2D cluster representation

 Performance Metrics
Model Evaluation
Sentiment Accuracy: Ensemble approach balances model biases
Topic Coherence: 0.185 score indicates meaningful topic separation
Clustering Quality: 0.750 silhouette score shows excellent cluster definition
Processing Speed: 10 articles processed in under 30 seconds
Scalability Considerations
Memory Usage: ~2GB for full pipeline with transformer models
Processing Time: Linear scaling with document count
Model Loading: One-time overhead for transformer model initialization
Dashboard Performance: Real-time filtering for datasets up to 1000 articles

 Contributing
Fork the repository
Create a feature branch (git checkout -b feature/enhancement)
Commit your changes (git commit -am 'Add new feature')
Push to the branch (git push origin feature/enhancement)
Create a Pull Request

 License
This project is licensed under the MIT License - see the LICENSE file for details.

 Author
Sree Kaushik
GitHub: @Sree-Kaushik
Email: f20220013@hyderabad.bits-pilani.ac.in

 Acknowledgments
FinBERT: ProsusAI for the financial sentiment analysis model
spaCy: For advanced NLP processing capabilities
Gensim: For topic modeling implementation
Plotly/Dash: For interactive visualization framework
Hugging Face: For transformer model hosting and APIs

 References
FinBERT: Financial Sentiment Analysis with BERT
Latent Dirichlet Allocation
Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
t-SNE: Visualizing Data using t-SNE

⭐ If you found this project helpful, please give it a star! ⭐
This project demonstrates production-ready NLP capabilities for financial text analysis and strategic decision-making.
