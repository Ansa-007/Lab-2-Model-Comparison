"""
Sentiment Analysis Lab Manual - State Graph Visualization
This lab demonstrates data flow and bottleneck identification in sentiment analysis
"""

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.datasets import make_classification
import numpy as np

def generate_sentiment_data():
    """Generate toy sentiment data for demonstration"""
    print("Generating sentiment dataset...")
    X, y = make_classification(n_samples=100, n_features=2, n_informative=1, 
                              n_redundant=0, n_repeated=0, n_classes=2, 
                              n_clusters_per_class=1, random_state=42)
    
    # Create text samples and labels
    texts = [f"sample {i}: {'pos' if label==1 else 'neg'}" for i, label in enumerate(y)]
    labels = ['positive' if l==1 else 'negative' for l in y]
    
    return X, y, texts, labels

def build_state_graph():
    """Build and visualize the state graph for sentiment analysis pipeline"""
    print("Building state graph...")
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes representing different data states
    G.add_nodes_from([
        "Raw Data (100 samples)", 
        "Preprocessed", 
        "Features", 
        "Model Input"
    ])
    
    # Add edges representing data transformations
    G.add_edges_from([
        ("Raw Data (100 samples)", "Preprocessed"),
        ("Preprocessed", "Features"),
        ("Features", "Model Input")
    ])
    
    return G

def visualize_graph(G):
    """Visualize the state graph"""
    plt.figure(figsize=(12, 8))
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Draw the graph with enhanced styling
    nx.draw(G, pos, 
            with_labels=True, 
            node_color='lightblue', 
            node_size=4000, 
            font_size=12, 
            font_weight='bold',
            arrows=True,
            arrowsize=20,
            edge_color='gray',
            width=2)
    
    plt.title("State Graph: Sentiment Analysis Pipeline - Identify Bottlenecks", 
              fontsize=16, fontweight='bold', pad=20)
    plt.show()

def analyze_class_balance(labels):
    """Analyze and report class balance in the dataset"""
    print("\n" + "="*50)
    print("CLASS BALANCE ANALYSIS")
    print("="*50)
    
    # Count classes
    negative_count = sum(1 for label in labels if label == 'negative')
    positive_count = sum(1 for label in labels if label == 'positive')
    total_count = len(labels)
    
    print(f"Total samples: {total_count}")
    print(f"Negative samples: {negative_count} ({negative_count/total_count*100:.1f}%)")
    print(f"Positive samples: {positive_count} ({positive_count/total_count*100:.1f}%)")
    
    # Check for imbalance
    imbalance_ratio = abs(negative_count - positive_count) / total_count
    if imbalance_ratio > 0.1:
        print(f"WARNING: IMBALANCE DETECTED: {imbalance_ratio*100:.1f}% difference")
        print("This could lead to biased model performance!")
    else:
        print("Dataset is well-balanced")
    
    print("="*50)
    
    return negative_count, positive_count

def identify_bottlenecks():
    """Identify potential bottlenecks in the sentiment analysis pipeline"""
    print("\n" + "="*50)
    print("POTENTIAL BOTTLENECKS")
    print("="*50)
    
    bottlenecks = [
        "1. Data Preprocessing: Text cleaning and normalization",
        "2. Feature Extraction: Converting text to numerical features", 
        "3. Class Imbalance: May affect model training",
        "4. Model Selection: Choosing appropriate algorithm",
        "5. Hyperparameter Tuning: Optimizing model performance"
    ]
    
    for bottleneck in bottlenecks:
        print(f"- {bottleneck}")
    
    print("="*50)

def main():
    """Main function to run the sentiment analysis lab"""
    print("SENTIMENT ANALYSIS LAB - STATE GRAPH VISUALIZATION")
    print("="*60)
    
    # Generate data
    X, y, texts, labels = generate_sentiment_data()
    
    # Analyze class balance
    negative_count, positive_count = analyze_class_balance(labels)
    
    # Build and visualize state graph
    G = build_state_graph()
    visualize_graph(G)
    
    # Identify potential bottlenecks
    identify_bottlenecks()
    
    print("\nLab completed successfully!")
    print("The state graph shows the flow of data through the sentiment analysis pipeline.")
    print("Use this visualization to identify where bottlenecks might occur in your process.")

if __name__ == "__main__":
    main()
