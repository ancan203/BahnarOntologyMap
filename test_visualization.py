import pandas as pd
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
from nltk.cluster import KMeansClusterer
import plotly.graph_objs as go

# Initialize SentenceTransformer model
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

# Define the Word class
class Word:
    def __init__(self, word: str, meaning: str) -> None:
        self.word = word
        self.meaning = meaning
        if len(meaning) == 0 or len(word) == 0:
            raise Exception("No empty strings allowed")
        if not isinstance(meaning, str):
            raise TypeError("Only strings are allowed")
        self.vector = model.encode(meaning)

    def get_word(self):
        return self.word

    def get_meaning(self):
        return self.meaning

    def get_vector(self):
        return self.vector

# Create Word objects
word_array = [
    Word("ERECH", "Small birds that cause a lot of damage in rice fields when the rice ripens."),
    Word("JÔMRANG", "Cockscomb, or other bird."),
    Word("JỞROL", "Pretty crested egret bird."),
    Word("BLUNG", "A type of fish."),
    Word("HLOR", "Very small river fish."),
    Word("XAKENG", "Kind of black fish."),
    Word("ANG", "Light, luminous, shine, shine."),
    Word("UNH", "Fire"),
    Word("PLA", "Flame, blade."),
    Word("XADRÂM", "The place of the river, the fountain, where we draw water for the household."),
    Word("XOK", "Small bay or cove in rivers."),
    Word("OR", "Low, wet ground near. watercourses.")
]

# Create a DataFrame
df = pd.DataFrame({"word": [word_object.word for word_object in word_array],
                   "meaning": [word_object.meaning for word_object in word_array],
                   "vector": [word_object.vector for word_object in word_array]})

# Define a function for clustering
def clustering_question(data, NUM_CLUSTERS=3):
    X = np.array(data['vector'].tolist())
    kclusterer = KMeansClusterer(
        NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance,
        repeats=25, avoid_empty_clusters=True)

    assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
    data['cluster'] = pd.Series(assigned_clusters, index=data.index)
    return data

# Cluster the data
results = clustering_question(df)

# Perform SVD for visualization
u, s, v = np.linalg.svd(np.array(df['vector'].tolist()), full_matrices=True)
flat_vectors = u[:, 0:2]

# Create a Plotly figure for visualization
fig = go.Figure()

# Define colors for clusters
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

# Create a list to store cluster names
cluster_names = ["Animal", "River", "Fire"]

# Add scatter plots for each word
for i in range(len(results)):
    word = results["word"].iloc[i]
    x, y = flat_vectors[i][0], flat_vectors[i][1]
    cluster = results["cluster"].iloc[i]
    color = colors[cluster]
    hover_text = f"Word: {word}<br>Definition: {results['meaning'].iloc[i]}"

    fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers+text', text=[word], textposition="bottom center",
                             marker=dict(size=12, color=color, opacity=0.8, line=dict(width=2, color='white')),
                             hovertext=[hover_text], name=cluster_names[cluster]))

# Set layout properties
fig.update_layout(
    title="Interactive Ontology Map",
    xaxis=dict(showticklabels=False, zeroline=False, showline=False),
    yaxis=dict(showticklabels=False, zeroline=False, showline=False),
    paper_bgcolor='lightgray',  # Background color
    plot_bgcolor='white',  # Plot background color
    font=dict(family='Arial', size=14),  # Font style and size
    legend=dict(x=0.02, y=0.98, bgcolor='white', bordercolor='black', borderwidth=1),  # Legend style
)

# Add interactivity to the plot
fig.update_layout(
    updatemenus=[
        dict(type="buttons",
             showactive=False,
             buttons=[dict(label="Drag Mode",
                           method="relayout",
                           args=["dragmode", "drag"]),
                      dict(label="Zoom Mode",
                           method="relayout",
                           args=["dragmode", "zoom"],
                           )
                      ]),
    ],
    showlegend=True,
    legend_title_text="Clusters"  # Legend title
)

fig.show()
