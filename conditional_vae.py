import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalVAE_3D(nn.Module):
    def __init__(self, num_classes, num_biomes, latent_dim=256, block_embed_dim=64, biome_embed_dim=16):
        super(ConditionalVAE_3D, self).__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes # 170 blocos
        self.num_biomes = num_biomes   # ~60-70?
        
        # --- CAMADAS DE EMBEDDING (Tradução de ID para Vetor) ---
        
        # Converte IDs de bloco (0-169) em vetores densos de 64 dimensões
        self.block_embedder = nn.Embedding(num_classes, block_embed_dim)
        
        # Converte IDs de bioma (0-N) em vetores densos de 16 dimensões
        self.biome_embedder = nn.Embedding(num_biomes, biome_embed_dim)
        
        # Input do Encoder será: (embed_bloco + embed_bioma)
        encoder_input_channels = block_embed_dim + biome_embed_dim # 64 + 16 = 80
        
        # Input do Decoder será: (dim_latente + embed_bioma)
        decoder_input_dim = latent_dim + biome_embed_dim # 256 + 16 = 272

        # --- 1. ENCODER (Comprime o Chunk + Bioma) ---
        # Input: (Batch, 80, 16, 32, 16)
        
        self.encoder_conv1 = nn.Sequential(
            nn.Conv3d(encoder_input_channels, 128, kernel_size=4, stride=2, padding=1), # -> (B, 128, 8, 16, 8)
            nn.ReLU(),
            nn.BatchNorm3d(128)
        )
        self.encoder_conv2 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1), # -> (B, 256, 4, 8, 4)
            nn.ReLU(),
            nn.BatchNorm3d(256)
        )
        self.encoder_conv3 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1), # -> (B, 512, 2, 4, 2)
            nn.ReLU(),
            nn.BatchNorm3d(512)
        )
        
        # Camada Flatten: 512 * 2 * 4 * 2 = 8192
        self.encoder_flat_dim = 512 * 2 * 4 * 2 
        
        # --- 2. BOTTLENECK (Onde sua observação acontece!) ---
        # Duas "cabeças" lineares para gerar os parâmetros da distribuição
        self.fc_mu = nn.Linear(self.encoder_flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_flat_dim, latent_dim)

        
        # --- 3. DECODER (Gera o Chunk a partir do Vetor Latente + Bioma) ---
        
        # Camada para "descomprimir" o vetor latente + bioma
        self.decoder_fc = nn.Linear(decoder_input_dim, self.encoder_flat_dim)
        
        self.decoder_deconv1 = nn.Sequential(
            # Input: (B, 512, 2, 4, 2)
            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1), # -> (B, 256, 4, 8, 4)
            nn.ReLU(),
            nn.BatchNorm3d(256)
        )
        self.decoder_deconv2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1), # -> (B, 128, 8, 16, 8)
            nn.ReLU(),
            nn.BatchNorm3d(128)
        )
        self.decoder_deconv3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1), # -> (B, 64, 16, 32, 16)
            nn.ReLU(),
            nn.BatchNorm3d(64)
        )
        
        # Camada final: Converte os 64 canais de volta para 170 (num_classes)
        # Este é o output de "logits", um para cada classe de bloco.
        self.decoder_output_conv = nn.Conv3d(64, num_classes, kernel_size=1, stride=1, padding=0)


    def encode(self, x_blocks, c_biome):
        """ Processa o chunk e o bioma pelo Encoder """
        
        # 1. Processa os blocos (IDs -> Embeddings)
        # Input x_blocks: (B, 16, 32, 16)
        x_embed = self.block_embedder(x_blocks) # -> (B, 16, 32, 16, 64)
        
        # Reordena para o formato do Conv3d: (B, C, D, H, W)
        x_embed = x_embed.permute(0, 4, 1, 2, 3) # -> (B, 64, 16, 32, 16)
        
        # 2. Processa o bioma (ID -> Embedding)
        # Input c_biome: (B,)
        c_embed = self.biome_embedder(c_biome) # -> (B, 16)
        
        # 3. Expande e concatena a condição (bioma)
        # Expande o embedding do bioma para ter o mesmo tamanho espacial do chunk
        # (B, 16) -> (B, 16, 1, 1, 1)
        c_embed_expanded = c_embed.view(-1, 16, 1, 1, 1)
        # (B, 16, 1, 1, 1) -> (B, 16, 16, 32, 16)
        c_embed_expanded = c_embed_expanded.expand(-1, -1, 16, 32, 16) 
        
        # Concatena os embeddings do chunk e do bioma
        # (B, 64, ...) + (B, 16, ...) -> (B, 80, 16, 32, 16)
        combined_input = torch.cat([x_embed, c_embed_expanded], dim=1)
        
        # 4. Passa pela rede convolucional
        h = self.encoder_conv1(combined_input)
        h = self.encoder_conv2(h)
        h = self.encoder_conv3(h)
        
        # 5. Achata (Flatten)
        h_flat = h.view(-1, self.encoder_flat_dim) # -> (B, 8192)
        
        # 6. Gera os parâmetros da distribuição
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        O TRUQUE DA REPARAMETRIZAÇÃO (O que você mencionou!)
        
        Amostra um vetor 'z' da distribuição N(mu, var) de uma
        forma que permite o backpropagation.
        z = mu + epsilon * sigma
        onde sigma = exp(0.5 * logvar)
             epsilon ~ N(0, 1)
        """
        std = torch.exp(0.5 * logvar) # Calcula o desvio padrão (sigma)
        eps = torch.randn_like(std)  # Gera ruído aleatório N(0, 1)
        
        return mu + eps * std # Retorna o vetor latente 'z' amostrado

    def decode(self, z, c_biome):
        """ Processa o vetor latente e o bioma pelo Decoder """
        
        # 1. Processa o bioma (ID -> Embedding)
        c_embed = self.biome_embedder(c_biome) # -> (B, 16)
        
        # 2. Concatena o vetor latente 'z' com a condição 'c'
        # (B, 256) + (B, 16) -> (B, 272)
        combined_z = torch.cat([z, c_embed], dim=1)
        
        # 3. Passa pela camada FC e "des-achata" (Unflatten)
        h = self.decoder_fc(combined_z) # -> (B, 8192)
        h = h.view(-1, 512, 2, 4, 2) # -> (B, 512, 2, 4, 2)
        
        # 4. Passa pela rede "de-convolucional"
        h = self.decoder_deconv1(h)
        h = self.decoder_deconv2(h)
        h = self.decoder_deconv3(h)
        
        # 5. Camada final de logits
        # h tem shape (B, 64, 16, 32, 16)
        # output_logits terá shape (B, 170, 16, 32, 16)
        output_logits = self.decoder_output_conv(h)
        
        return output_logits

    def forward(self, x_blocks, c_biome):
        """
        O fluxo completo: Encode -> Reparameterize -> Decode
        """
        # x_blocks: (B, 16, 32, 16)
        # c_biome: (B,)
        
        mu, logvar = self.encode(x_blocks, c_biome)
        
        z = self.reparameterize(mu, logvar)
        
        reconstructed_logits = self.decode(z, c_biome)
        
        # Retorna os logits da reconstrução, e também
        # mu e logvar, pois precisaremos deles para
        # calcular a perda (loss) da VAE.
        return reconstructed_logits, mu, logvar

def vae_loss_function(reconstructed_logits, target_blocks, mu, logvar, beta=1.0):
    # reduction='sum': Soma o erro de todos os 8192 voxels de todos os chunks do batch
    recon_loss = F.cross_entropy(reconstructed_logits, target_blocks, reduction='sum')
    
    # Normalizamos pelo batch_size para não explodir se mudarmos o tamanho do batch
    batch_size = target_blocks.size(0)
    recon_loss = recon_loss / batch_size
    
    # KLD permanece a mesma lógica (mas agora as magnitudes são comparáveis)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld_loss = kld_loss / batch_size
    
    total_loss = recon_loss + (beta * kld_loss)
    return total_loss, recon_loss, kld_loss