# verificar_chunks.py
import json
with open("app/rag_data/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"Total de chunks: {len(chunks)}")