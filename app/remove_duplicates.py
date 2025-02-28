import json
import os

# Configurações
INPUT_JSON = "app/rag_data/chunks.json"
OUTPUT_JSON = "app/rag_data/chunks_unique.json"

def remove_duplicates(input_file, output_file):
    # Carregar os chunks do arquivo de entrada
    if not os.path.exists(input_file):
        print(f"Arquivo {input_file} não encontrado.")
        return
    
    with open(input_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print(f"Total de chunks carregados: {len(chunks)}")

    # Remover duplicatas mantendo a ordem original
    unique_chunks = list(dict.fromkeys(chunks))
    
    print(f"Total de chunks únicos: {len(unique_chunks)}")
    print(f"Duplicatas removidas: {len(chunks) - len(unique_chunks)}")

    # Salvar os chunks únicos no arquivo de saída
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(unique_chunks, f, ensure_ascii=False, indent=4)
    
    print(f"Chunks únicos salvos em {output_file}")

if __name__ == "__main__":
    remove_duplicates(INPUT_JSON, OUTPUT_JSON)