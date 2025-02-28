# visualize_embeddings.py

import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial import distance_matrix

def compute_density(df, radius=0.05):
    """
    Calcula a densidade local para cada ponto, contando quantos pontos estão a menos de 'radius'.
    """
    coords = df[['x', 'y', 'z']].values
    dist_mat = distance_matrix(coords, coords)
    density = np.sum(dist_mat < radius, axis=1) - 1  # subtrai o próprio ponto
    return density

def main():
    map_file = "map_data.csv"
    summary_file = "cluster_summaries.json"

    if not os.path.exists(map_file):
        print(f"[ERRO] Arquivo {map_file} não encontrado.")
        return
    if not os.path.exists(summary_file):
        print(f"[ERRO] Arquivo {summary_file} não encontrado.")
        return

    print("[INFO] Carregando dados do mapa...")
    df = pd.read_csv(map_file)
    
    print("[INFO] Carregando resumos dos clusters...")
    with open(summary_file, "r", encoding="utf-8") as f:
        cluster_summaries = json.load(f)
    
    # Calcula densidade para ajustar tamanho de fonte
    density = compute_density(df, radius=0.05)
    d_min, d_max = density.min(), density.max()
    if d_max > d_min:
        density_norm = (density - d_min) / (d_max - d_min)
    else:
        density_norm = np.zeros_like(density)

    # Tamanho mínimo e máximo da fonte
    font_size_min = 8
    font_size_max = 20
    font_sizes = font_size_min + (font_size_max - font_size_min) * density_norm

    # Definimos 5 cores para os clusters
    cluster_colors = {
        0: "red",
        1: "green",
        2: "blue",
        3: "orange",
        4: "purple"
    }

    # Cria o gráfico 3D com traces separados para cada cluster
    fig = go.Figure()
    unique_clusters = sorted(df["cluster"].unique())
    for cluster_id in unique_clusters:
        cluster_data = df[df["cluster"] == cluster_id].copy()
        cidx = cluster_data.index
        sub_font_sizes = font_sizes[cidx]
        color = cluster_colors.get(cluster_id, "black")
        fig.add_trace(
            go.Scatter3d(
                x=cluster_data["x"],
                y=cluster_data["y"],
                z=cluster_data["z"],
                mode="text",
                text=cluster_data["keyword"],
                textfont=dict(
                    color=color,
                    size=sub_font_sizes
                ),
                hovertext=cluster_data["chunk_text"],
                hoverinfo="text",
                name=f"Cluster {cluster_id}"
            )
        )

    # Configura o layout: fundo claro e eixos com grid claro
    fig.update_layout(
        scene=dict(
            xaxis=dict(backgroundcolor="whitesmoke", gridcolor="lightgray", color="black"),
            yaxis=dict(backgroundcolor="whitesmoke", gridcolor="lightgray", color="black"),
            zaxis=dict(backgroundcolor="whitesmoke", gridcolor="lightgray", color="black"),
            bgcolor="whitesmoke"
        ),
        paper_bgcolor="whitesmoke",
        font=dict(color="black"),
        title="Visualização 3D dos Embeddings com Keywords coloridas por Cluster"
    )
    
    # Calcula os centróides dos clusters para anotações
    centroid_x, centroid_y, centroid_z, annotation_text = [], [], [], []
    for cluster_id in unique_clusters:
        cluster_data = df[df["cluster"] == cluster_id]
        centroid = cluster_data[["x", "y", "z"]].mean().to_dict()
        if len(cluster_data) == 1:
            jitter = np.random.uniform(-0.05, 0.05)
            centroid["x"] += jitter
            centroid["y"] += jitter
            centroid["z"] += jitter
        centroid_x.append(centroid["x"])
        centroid_y.append(centroid["y"])
        centroid_z.append(centroid["z"])
        summary = cluster_summaries.get(str(cluster_id), cluster_summaries.get(cluster_id, "N/A"))
        annotation_text.append(f"Cluster {cluster_id}: {summary}")

    fig.add_trace(
        go.Scatter3d(
            x=centroid_x,
            y=centroid_y,
            z=centroid_z,
            mode="text",
            text=annotation_text,
            textfont=dict(color="darkblue", size=14),
            textposition="top center",
            showlegend=False
        )
    )
    
    fig.show()
    print("[INFO] Gráfico exibido.")

if __name__ == "__main__":
    main()
