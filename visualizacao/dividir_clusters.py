# dividir_clusters.py

import os
import pandas as pd

def save_cluster_texts(cluster_id: int, cluster_texts: list):
    """
    Salva os textos de um cluster em um arquivo separado.
    """
    filename = f"cluster_{cluster_id}_texts.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n\n".join(cluster_texts))

def main():
    map_file = "map_data.csv"
    if not os.path.exists(map_file):
        print(f"[ERRO] Arquivo {map_file} não encontrado em {os.getcwd()}")
        return

    df = pd.read_csv(map_file)
    n_clusters = df["cluster"].nunique()

    for cluster_id in range(n_clusters):
        cluster_texts = df[df["cluster"] == cluster_id]["chunk_text"].tolist()
        save_cluster_texts(cluster_id, cluster_texts)

if __name__ == "__main__":
    main()