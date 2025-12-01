import torch
from torch.utils.data import Dataset

class MinecraftChunkDataset(Dataset):
    """
    Carrega o dataset 'processed_dataset.pt' que criamos.
    Este dataset já contém os tensores de inteiros (IDs de blocos).
    """

    def __init__(self, pt_file_path):
        try:
            data_dict = torch.load(pt_file_path, map_location="cpu", weights_only=False)

            self.blocks = data_dict["blocks"]
            self.biomes = data_dict["biomes"]

            self.num_chunks = data_dict["num_chunks"]
            self.num_unique_blocks = data_dict["num_unique_blocks"]

            self.num_biomes = self.biomes.max().item() + 1

            print(f"Dataset carregado: {self.num_chunks} chunks.")
            print(f"Número de blocos únicos (classes): {self.num_unique_blocks}")
            print(f"Número de biomas únicos (condições): {self.num_biomes}")

        except FileNotFoundError:
            print(f"ERRO: Arquivo processado não encontrado em: {pt_file_path}")
            print("Por favor, rode o script de pré-processamento primeiro.")
            raise
        except KeyError:
            print(
                "ERRO: O arquivo .pt não contém as chaves esperadas ('blocks', 'biomes')."
            )
            raise

    def __len__(self):
        """Retorna o número total de amostras no dataset."""
        return self.num_chunks

    def __getitem__(self, idx):
        """Retorna uma única amostra (chunk e seu bioma)."""
        # Retorna o chunk de IDs de blocos e o ID do bioma correspondente
        return self.blocks[idx], self.biomes[idx]