# Config Process Dataset
RAW_DATA_FILE = 'data/minecraft_voxel_dataset.npz'

OUTPUT_PT_FILE = 'data/processed_dataset.pt'
MAP_BLOCK_TO_ID_FILE = 'data/block_to_id.json'
MAP_ID_TO_BLOCK_FILE = 'data/id_to_block.json'


# Config Training
PT_FILE_PATH = 'data/processed_dataset.pt'
LOSS_PLOT_PATH = 'media/vae_loss_curve.png'

EPOCHS = 50 
BATCH_SIZE = 32
LEARNING_RATE = 1e-4 # Uma taxa de aprendizado segura para VAEs
LATENT_DIM = 256     # Dimensão do bottleneck (vetor z)
BLOCK_EMBED_DIM = 64 # Dimensão do embedding de blocos
BIOME_EMBED_DIM = 16 # Dimensão do embedding de biomas
BETA_KLD_WEIGHT = 0.5  # Peso para a perda KL (beta). Começar baixo é bom.


# Config Generate Samples
CHECKPOINT_PATH = 'models/best_vae_model.pt'
DATASET_INFO_PATH = 'data/processed_dataset.pt' # Para pegar nº de classes e biomas
ID_TO_BLOCK_PATH = 'data/id_to_block.json'
OUTPUT_DIR = 'generated_samples'
NUM_SAMPLES = 50

CHUNK_X, CHUNK_Y, CHUNK_Z = 16, 32, 16