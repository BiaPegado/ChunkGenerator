import numpy as np
import torch
import json
import os
from tqdm import tqdm
from config import RAW_DATA_FILE, OUTPUT_PT_FILE, MAP_BLOCK_TO_ID_FILE, MAP_ID_TO_BLOCK_FILE

def preprocess():
    
    print(f"Iniciando pré-processamento de '{RAW_DATA_FILE}'...")
    try:
        raw_data_npz = np.load(RAW_DATA_FILE, allow_pickle=True)
        blocks_data = raw_data_npz['blocks']
        biomes_data = raw_data_npz['biomes']
        print(f"Dataset bruto carregado. Shape dos blocos: {blocks_data.shape}")
    except FileNotFoundError:
        print(f"ERRO: Arquivo de entrada não encontrado: {RAW_DATA_FILE}")
        print("Por favor, baixe o dataset e coloque-o no mesmo diretório.")
        return
    except Exception as e:
        print(f"Ocorreu um erro ao carregar o .npz: {e}")
        return

    print("Passo 1/4: Escaneando todos os chunks para encontrar blocos únicos...")
    
    unique_blocks = set()

    for i in tqdm(range(blocks_data.shape[0]), desc="Escaninhando chunks"):
        chunk = blocks_data[i]
        unique_blocks.update(np.unique(chunk))

    sorted_unique_blocks = sorted(list(unique_blocks))
    
    num_unique_blocks = len(sorted_unique_blocks)
    print(f"Escaninhamento completo. Encontrados {num_unique_blocks} tipos de blocos únicos.")

    print("Passo 2/4: Criando e salvando mapas de tradução (JSON)...")

    block_to_id = {block_name: i for i, block_name in enumerate(sorted_unique_blocks)}
    
    id_to_block = {i: block_name for block_name, i in block_to_id.items()}

    with open(MAP_BLOCK_TO_ID_FILE, 'w', encoding='utf-8') as f:
        json.dump(block_to_id, f, indent=4) 

    with open(MAP_ID_TO_BLOCK_FILE, 'w', encoding='utf-8') as f:
        json.dump(id_to_block, f, indent=4)

    print(f"Mapas salvos em '{MAP_BLOCK_TO_ID_FILE}' e '{MAP_ID_TO_BLOCK_FILE}'.")

    print("Passo 3/4: Traduzindo o dataset para IDs numéricos...")

    translated_blocks = np.zeros(blocks_data.shape, dtype=np.int16)

    for i in tqdm(range(blocks_data.shape[0]), desc="Traduzindo chunks"):
        chunk_str = blocks_data[i]
        
        flat_chunk_str = chunk_str.flatten()
        
        flat_chunk_int = [block_to_id[block_name] for block_name in flat_chunk_str]
        
        translated_blocks[i] = np.array(flat_chunk_int, dtype=np.int16).reshape(chunk_str.shape)

    print("Tradução completa.")

    print("Passo 4/4: Convertendo para Tensores PyTorch e salvando...")

    blocks_tensor = torch.from_numpy(translated_blocks).long()
    
    biomes_tensor = torch.from_numpy(biomes_data).long()

    processed_data_dict = {
        'blocks': blocks_tensor,
        'biomes': biomes_tensor,
        'num_chunks': blocks_tensor.shape[0],
        'num_unique_blocks': num_unique_blocks
    }
    
    torch.save(processed_data_dict, OUTPUT_PT_FILE)
    
    print("-" * 30)
    print("✨ Pré-processamento CONCLUÍDO! ✨")
    print(f"Dados processados salvos em: '{OUTPUT_PT_FILE}'")
    print("-" * 30)

    raw_data_npz.close()


if __name__ == "__main__":
    if os.path.exists(OUTPUT_PT_FILE):
        print(f"O arquivo processado '{OUTPUT_PT_FILE}' já existe.")
        print("Pulando pré-processamento. Delete o arquivo para rodar novamente.")
    else:
        preprocess()
