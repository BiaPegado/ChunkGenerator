import numpy as np
import os 

NPZ_FILE = 'data/minecraft_voxel_dataset.npz'


def inspect_npz(filepath):
    """Carrega e inspeciona o arquivo .npz."""
    print("--- Inspecionando o arquivo NumPy (.npz) ---")
    
    if not os.path.exists(filepath):
        print(f"ERRO: Arquivo não encontrado: {filepath}\n")
        return

    try:
        npz_data = np.load(filepath, allow_pickle=True)
        
        print(f"Arquivos (chaves) dentro do .npz: {npz_data.files}")
        
        if 'blocks' in npz_data:
            blocks_array = npz_data['blocks']
            print(f"  [blocks] Shape do array: {blocks_array.shape}")
            print(f"  [blocks] Tipo de dado (dtype): {blocks_array.dtype}")
        
        if 'biomes' in npz_data:
            biomes_array = npz_data['biomes']
            print(f"  [biomes] Shape do array: {biomes_array.shape}")
            print(f"  [biomes] Tipo de dado (dtype): {biomes_array.dtype}")

        if 'num_chunks' in npz_data:
            # Pode ser um array de 0 dimensões, por isso o .item()
            print(f"  [num_chunks] Total: {npz_data['num_chunks'].item()}")

        npz_data.close()
        
    except Exception as e:
        print(f"Ocorreu um erro ao ler o .npz: {e}")


if __name__ == "__main__":
    inspect_npz(NPZ_FILE)
