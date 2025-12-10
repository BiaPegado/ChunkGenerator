import torch
import json
import os
import numpy as np
from tqdm import tqdm

from conditional_vae import ConditionalVAE_3D
from config import CHECKPOINT_PATH, DATASET_INFO_PATH, ID_TO_BLOCK_PATH, OUTPUT_DIR, NUM_SAMPLES, CHUNK_X, CHUNK_Y, CHUNK_Z



CENTER_X = CHUNK_X // 2
CENTER_Y = CHUNK_Y // 2
CENTER_Z = CHUNK_Z // 2

def generate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    print("Carregando metadados do dataset...")
    if not os.path.exists(DATASET_INFO_PATH):
        print(f"ERRO: '{DATASET_INFO_PATH}' não encontrado. Precisamos dele para saber o num_classes.")
        return

    data_info = torch.load(DATASET_INFO_PATH, map_location='cpu', weights_only=False)
    if isinstance(data_info, dict) and 'num_unique_blocks' in data_info:
        num_classes = data_info['num_unique_blocks']
        if 'biomes' in data_info:
            num_biomes = data_info['biomes'].max().item() + 1
        else:
            print("AVISO: Número de biomas não encontrado, usando valor estimado (verifique seu modelo).")
            num_biomes = 70 
    else:
        print("Formato do dataset processado desconhecido.")
        return

    # Garante que num_biomas esteja definido antes de prosseguir
    if 'num_biomas' not in locals():
        num_biomas = 2  # Valor padrão caso não tenha sido definido

    print(f"Configuração do Modelo detectada: {num_classes} blocos, {num_biomas} biomas.")

    with open(ID_TO_BLOCK_PATH, 'r') as f:
        raw_dict = json.load(f)
        id_to_block = {int(k): v for k, v in raw_dict.items()}

    print("Carregando modelo treinado...")
    model = ConditionalVAE_3D(
        num_classes=num_classes,
        num_biomes=num_biomas,
        latent_dim=256,
        block_embed_dim=64,
        biome_embed_dim=16
    ).to(device)

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    TEMPERATURE = 0.8

    print(f"Gerando {NUM_SAMPLES} amostras com Temperatura {TEMPERATURE}...")

    for i in tqdm(range(NUM_SAMPLES)):

        # Amostra um vetor latente z da distribuição normal padrão N(0, 1)
        z = torch.randn(1, 256).to(device)
    
        # Seleciona aleatoriamente um bioma para condicionar a geração
        rand_biome = torch.randint(0, num_biomes, (1,)).to(device)
        
        with torch.no_grad():
            # Decodifica o vetor latente z e o bioma para gerar logits (valores não normalizados)
            logits = model.decode(z, rand_biome)
        
        # Ajusta a temperatura para controlar a aleatoriedade da amostragem
        logits = logits / TEMPERATURE
        
        # Aplica softmax para converter logits em probabilidades
        probs = torch.nn.functional.softmax(logits, dim=1)
        
        # Reorganiza as dimensões para facilitar a amostragem
        permuted_probs = probs.permute(0, 2, 3, 4, 1) 
        
        # Achata as probabilidades para realizar a amostragem
        flat_probs = permuted_probs.reshape(-1, num_classes)
        
        # Realiza a amostragem multinomial para obter índices gerados
        flat_indices = torch.multinomial(flat_probs, num_samples=1)
        
        # Reconstrói os índices gerados na forma original do chunk
        generated_indices = flat_indices.reshape(16, 32, 16).cpu().numpy()
        
        sparse_list = []
        
        it = np.nditer(generated_indices, flags=['multi_index'])
        for block_id in it:
            x, y, z = it.multi_index
            block_id = int(block_id)
            
            block_name = id_to_block.get(block_id, "unknown")
            
            # Ignora blocos de "ar" e desconhecidos para otimizar o armazenamento
            if "air" not in block_name and block_name != "unknown":
                dx = x - CENTER_X
                dy = y - CENTER_Y
                dz = z - CENTER_Z
                
                block_data = {
                    "id": block_name,
                    "dx": dx,
                    "dy": dy,
                    "dz": dz
                }
                sparse_list.append(block_data)

        # Salva a amostra gerada em um arquivo JSON
        filename = os.path.join(OUTPUT_DIR, f"sample_{i+1:02d}.txt")
        with open(filename, 'w') as f:
            json.dump(sparse_list, f) 

    print("\nConcluído!")

if __name__ == "__main__":
    generate()