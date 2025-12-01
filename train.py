import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from conditional_vae import ConditionalVAE_3D, vae_loss_function
from config import (
    PT_FILE_PATH,
    LOSS_PLOT_PATH,
    EPOCHS,
    BATCH_SIZE,
    BETA_KLD_WEIGHT,
    BIOME_EMBED_DIM,
    BLOCK_EMBED_DIM,
    LEARNING_RATE,
    LATENT_DIM,
)
from dataset import MinecraftChunkDataset


def save_loss_plot(train_losses, recon_losses, kld_losses, filename="loss_curve.png"):
    """
    Salva um gráfico das curvas de perda.
    """
    plt.figure(figsize=(10, 6))

    # Plota a perda total
    plt.plot(train_losses, label="Perda Total (Total Loss)", color="blue")

    # Plota as componentes da perda
    plt.plot(
        recon_losses,
        label="Perda de Reconstrução (Recon)",
        color="green",
        linestyle="--",
    )
    plt.plot(kld_losses, label="Perda KL (KLD)", color="red", linestyle="--")

    plt.title("Curvas de Perda do Treinamento da VAE")
    plt.xlabel("Época (Epoch)")
    plt.ylabel("Perda (Loss)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(filename)
    print(f"Gráfico de perda salvo em '{filename}'")

    plt.close()
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    print("Carregando dataset...")
    try:
        dataset = MinecraftChunkDataset(PT_FILE_PATH)

        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
    except Exception as e:
        print(f"Erro ao carregar o dataset: {e}")
        exit()

    # --- 2. Instanciar Modelo e Otimizador ---
    print("Construindo modelo...")
    model = ConditionalVAE_3D(
        num_classes=dataset.num_unique_blocks,  # 170
        num_biomes=dataset.num_biomes,  # ~N (dataset nos diz)
        latent_dim=LATENT_DIM,
        block_embed_dim=BLOCK_EMBED_DIM,
        biome_embed_dim=BIOME_EMBED_DIM,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history_total_loss = []
    history_recon_loss = []
    history_kld_loss = []

    print(f"Iniciando treinamento por {EPOCHS} épocas...")
    print("-" * 30)

    best_loss = float("inf")  # Começa com infinito para qualquer loss ser menor

    print(f"Iniciando treinamento por {EPOCHS} épocas...")
    print("-" * 30)

    for epoch in range(EPOCHS):
        model.train()

        epoch_loss_total = 0.0
        epoch_loss_recon = 0.0
        epoch_loss_kld = 0.0

        for batch_blocks, batch_biomes in tqdm(
            dataloader, desc=f"Época {epoch + 1}/{EPOCHS}"
        ):
            blocks = batch_blocks.to(device)
            biomes = batch_biomes.to(device)

            reconstructed_logits, mu, logvar = model(blocks, biomes)

            total_loss, recon_loss, kld_loss = vae_loss_function(
                reconstructed_logits, blocks, mu, logvar, beta=BETA_KLD_WEIGHT
            )

            optimizer.zero_grad()
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss_total += total_loss.item()
            epoch_loss_recon += recon_loss.item()
            epoch_loss_kld += kld_loss.item()

        avg_loss_total = epoch_loss_total / len(dataloader)
        avg_loss_recon = epoch_loss_recon / len(dataloader)
        avg_loss_kld = epoch_loss_kld / len(dataloader)

        history_total_loss.append(avg_loss_total)
        history_recon_loss.append(avg_loss_recon)
        history_kld_loss.append(avg_loss_kld)

        print(f"\nFim da Época {epoch + 1}")
        print(f"  Perda Total Média: {avg_loss_total:.4f}")

        if avg_loss_total < best_loss:
            best_loss = avg_loss_total
            torch.save(model.state_dict(), "models/best_vae_model.pt")
            print(f"🌟 NOVO RECORDE! Melhor modelo salvo (Loss: {best_loss:.4f})")
        else:
            print(f"  (Não superou o recorde de {best_loss:.4f})")

        torch.save(model.state_dict(), "models/latest_vae_model.pt")
        print("-" * 30)

    save_loss_plot(
        history_total_loss, history_recon_loss, history_kld_loss, LOSS_PLOT_PATH
    )

    print("Treinamento concluído!")
