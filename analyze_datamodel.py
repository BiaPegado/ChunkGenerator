# -*- coding: utf-8 -*-
"""
Script de Inspeção (Corrigido)

Adiciona as flags 'allow_pickle=True' (NumPy) e 'weights_only=False' (PyTorch)
para permitir o carregamento de arquivos que contêm objetos 'pickle',
o que é comum em datasets salvos com versões mais antigas.
"""

import numpy as np
import torch
import os 

# --- Configuração ---
NPZ_FILE = 'minecraft_voxel_dataset.npz'
PT_FILE = 'minecraft_voxel_dataset.pt'
# --------------------


def inspect_npz(filepath):
    """Carrega e inspeciona o arquivo .npz."""
    print(f"--- Inspecionando o arquivo NumPy (.npz) ---")
    
    if not os.path.exists(filepath):
        print(f"ERRO: Arquivo não encontrado: {filepath}\n")
        return

    try:
        # --- CORREÇÃO AQUI ---
        # O erro 'Object arrays cannot be loaded' acontece porque
        # o NumPy, por segurança, desabilitou o 'pickle' por padrão.
        # Ao definir 'allow_pickle=True', nós dizemos ao NumPy:
        # "Eu confio neste arquivo, pode carregar os objetos."
        npz_data = np.load(filepath, allow_pickle=True)
        # ---------------------
        
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


def inspect_pt(filepath):
    """Carrega e inspeciona o arquivo .pt."""
    print(f"\n--- Inspecionando o arquivo PyTorch (.pt) ---")

    if not os.path.exists(filepath):
        print(f"ERRO: Arquivo não encontrado: {filepath}\n")
        return

    try:
        # --- CORREÇÃO AQUI ---
        # O erro 'Weights only load failed' é o mesmo problema de segurança
        # do NumPy. A partir do PyTorch 2.6, 'weights_only' é 'True' por padrão.
        # O erro 'Unsupported global: GLOBAL numpy...' confirma que o arquivo
        # contém objetos NumPy (e não apenas tensores PyTorch).
        # Ao definir 'weights_only=False', nós dizemos ao PyTorch:
        # "Eu confio neste arquivo, pode 'desempacotar' (unpickle) os objetos."
        pt_data = torch.load(filepath, map_location='cpu', weights_only=False)
        # ---------------------
        
        print(f"Tipo de objeto carregado: {type(pt_data)}")
        
        if isinstance(pt_data, dict):
            print(f"Chaves do dicionário: {pt_data.keys()}")
            
            if 'blocks' in pt_data:
                blocks_tensor = pt_data['blocks']
                print(f"  [blocks] Shape do tensor: {blocks_tensor.shape}")
                print(f"  [blocks] Tipo de dado (dtype): {blocks_tensor.dtype}")
                print(f"  [blocks] Dispositivo: {blocks_tensor.device}")
            
            if 'biomes' in pt_data:
                biomes_tensor = pt_data['biomes']
                print(f"  [biomes] Shape do tensor: {biomes_tensor.shape}")
                print(f"  [biomes] Tipo de dado (dtype): {biomes_tensor.dtype}")
        
        else:
            print("O arquivo .pt não é um dicionário, inspecione manualmente.")
            
    except Exception as e:
        print(f"Ocorreu um erro ao ler o .pt: {e}")


# --- Execução Principal ---
if __name__ == "__main__":
    inspect_npz(NPZ_FILE)
    inspect_pt(PT_FILE)
