import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, date
import dash_bootstrap_components as dbc

class FinancialNLPDashboard:
    def __init__(self, df):
        self.df = df
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("Financial News NLP Analytics Dashboard", 
                           className="text-center mb-4",
                           style={'color': '#2c3e50', 'fontWeight': 'bold'})
                ])
            ]),
            
            # Summary Cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{len(self.df)}", className="card-title"),
                            html.P("Total Articles", className="card-text")
                        ])
                    ], color="primary", outline=True)
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{self.df['vader_sentiment'].value_counts().get('positive', 0)}", 
                                   className="card-title"),
                            html.P("Positive Sentiment", className="card-text")
                        ])
                    ], color="success", outline=True)
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{self.df['vader_sentiment'].value_counts().get('negative', 0)}", 
                                   className="card-title"),
                            html.P("Negative Sentiment", className="card-text")
                        ])
                    ], color="danger", outline=True)
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{len(self.df['cluster'].unique())}", 
                                   className="card-title"),
                            html.P("Topic Clusters", className="card-text")
                        ])
                    ], color="info", outline=True)
                ], width=3)
            ], className="mb-4"),
            
            # Filters
            dbc.Row([
                dbc.Col([
                    html.Label("Select Sentiment:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='sentiment-filter',
                        options=[
                            {'label': 'All', 'value': 'all'},
                            {'label': 'Positive', 'value': 'positive'},
                            {'label': 'Negative', 'value': 'negative'},
                            {'label': 'Neutral', 'value': 'neutral'}
                        ],
                        value='all',
                        clearable=False
                    )
                ], width=4),
                
                dbc.Col([
                    html.Label("Select Cluster:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='cluster-filter',
                        options=[{'label': f'Cluster {i}', 'value': i} 
                                for i in sorted(self.df['cluster'].unique())] + 
                               [{'label': 'All', 'value': 'all'}],
                        value='all',
                        clearable=False
                    )
                ], width=4),
                
                dbc.Col([
                    html.Label("Select Source:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='source-filter',
                        options=[{'label': source, 'value': source} 
                                for source in self.df['source'].unique()] + 
                               [{'label': 'All', 'value': 'all'}],
                        value='all',
                        clearable=False
                    )
                ], width=4)
            ], className="mb-4"),
            
            # Main Charts
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='sentiment-distribution')
                ], width=6),
                
                dbc.Col([
                    dcc.Graph(id='topic-distribution')
                ], width=6)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='cluster-visualization')
                ], width=12)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='sentiment-timeline')
                ], width=6),
                
                dbc.Col([
                    dcc.Graph(id='source-analysis')
                ], width=6)
            ], className="mb-4"),
            
            # Data Table
            dbc.Row([
                dbc.Col([
                    html.H3("Article Details", style={'fontWeight': 'bold'}),
                    html.Div(id='data-table')
                ], width=12)
            ])
            
        ], fluid=True)
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output('sentiment-distribution', 'figure'),
             Output('topic-distribution', 'figure'),
             Output('cluster-visualization', 'figure'),
             Output('sentiment-timeline', 'figure'),
             Output('source-analysis', 'figure'),
             Output('data-table', 'children')],
            [Input('sentiment-filter', 'value'),
             Input('cluster-filter', 'value'),
             Input('source-filter', 'value')]
        )
        def update_dashboard(sentiment_filter, cluster_filter, source_filter):
            # Filter data
            filtered_df = self.df.copy()
            
            if sentiment_filter != 'all':
                filtered_df = filtered_df[filtered_df['vader_sentiment'] == sentiment_filter]
            
            if cluster_filter != 'all':
                filtered_df = filtered_df[filtered_df['cluster'] == cluster_filter]
            
            if source_filter != 'all':
                filtered_df = filtered_df[filtered_df['source'] == source_filter]
            
            # Sentiment Distribution
            sentiment_counts = filtered_df['vader_sentiment'].value_counts()
            sentiment_fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Distribution",
                color_discrete_map={
                    'positive': '#2ecc71',
                    'negative': '#e74c3c',
                    'neutral': '#95a5a6'
                }
            )
            
            # Topic Distribution
            if 'dominant_topic' in filtered_df.columns:
                topic_counts = filtered_df['dominant_topic'].value_counts()
                topic_fig = px.bar(
                    x=topic_counts.index,
                    y=topic_counts.values,
                    title="Topic Distribution",
                    labels={'x': 'Topic', 'y': 'Count'}
                )
            else:
                topic_fig = px.bar(title="Topic Distribution - No Data Available")
            
            # Cluster Visualization (t-SNE)
            if 'tsne_x' in filtered_df.columns and 'tsne_y' in filtered_df.columns:
                cluster_fig = px.scatter(
                    filtered_df,
                    x='tsne_x',
                    y='tsne_y',
                    color='cluster',
                    hover_data=['headline', 'vader_sentiment'],
                    title="Document Clusters (t-SNE Visualization)",
                    labels={'tsne_x': 't-SNE Component 1', 'tsne_y': 't-SNE Component 2'}
                )
            else:
                cluster_fig = px.scatter(title="Cluster Visualization - No Data Available")
            
            # Sentiment Timeline
            if 'date' in filtered_df.columns:
                timeline_data = filtered_df.groupby(['date', 'vader_sentiment']).size().reset_index(name='count')
                timeline_fig = px.line(
                    timeline_data,
                    x='date',
                    y='count',
                    color='vader_sentiment',
                    title="Sentiment Trends Over Time",
                    color_discrete_map={
                        'positive': '#2ecc71',
                        'negative': '#e74c3c',
                        'neutral': '#95a5a6'
                    }
                )
            else:
                timeline_fig = px.line(title="Sentiment Timeline - No Date Data Available")
            
            # Source Analysis
            source_sentiment = filtered_df.groupby(['source', 'vader_sentiment']).size().reset_index(name='count')
            source_fig = px.bar(
                source_sentiment,
                x='source',
                y='count',
                color='vader_sentiment',
                title="Sentiment by News Source",
                color_discrete_map={
                    'positive': '#2ecc71',
                    'negative': '#e74c3c',
                    'neutral': '#95a5a6'
                }
            )
            
            # Data Table
            table_data = filtered_df[['headline', 'vader_sentiment', 'cluster', 'source', 'date']].head(10)
            
            table = dbc.Table.from_dataframe(
                table_data,
                striped=True,
                bordered=True,
                hover=True,
                responsive=True,
                size='sm'
            )
            
            return sentiment_fig, topic_fig, cluster_fig, timeline_fig, source_fig, table
    
    def run_server(self, debug=True, port=8050):
        """Run the dashboard server"""
        print(f"Starting dashboard server on http://localhost:{port}")
        # Fixed: Use app.run instead of app.run_server
        self.app.run(debug=debug, port=port, host='127.0.0.1')

if __name__ == "__main__":
    # Load processed data (this would be your actual processed dataframe)
    from data_preprocessing import TextPreprocessor, load_sample_data
    from sentiment_analysis import SentimentAnalyzer
    from topic_modeling import TopicModeler
    from clustering import TextClusterer
    
    # Create sample processed data
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
    
    # Add clustering
    clusterer = TextClusterer()
    embeddings = clusterer.create_embeddings(topic_df['processed_text'].tolist())
    cluster_labels = clusterer.perform_clustering(embeddings, n_clusters=3)
    topic_df['cluster'] = cluster_labels
    
    # Add t-SNE coordinates for visualization
    tsne_results = clusterer.perform_tsne(embeddings)
    topic_df['tsne_x'] = tsne_results[:, 0]
    topic_df['tsne_y'] = tsne_results[:, 1]
    
    # Create and run dashboard
    dashboard = FinancialNLPDashboard(topic_df)
    dashboard.run_server(debug=True, port=8050)
