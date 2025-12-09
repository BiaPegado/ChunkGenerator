# Projeto final de IA Generativa

## Componentes

* Artur Revorêdo Pinto
* Beatriz Pegado

## Introdução

O projeto tem como objetivo treinar um __variational autoencoder__ a partir de __voxels__ que representam __chunks__ do jogo Minecraft no formato 16x32x16 para conseguir gerar novas instâncias de __chunk__.

## Como executar

Colete no [kaggle](https://www.kaggle.com/datasets/zwniff/mc-chunks) o arquivo ``minecraft_voxel_dataset.npz``, adicione esse arquivo na pasta data e rode o script ``pre_process.py`` com o uv:
```bash
# mkdir para garantir criação de pastas
mkdir -p models
mkdir -p media
uv run pre_process.py
```

Se você fez tudo certo até agora, você tem um arquivo ``prcessed_dataset.pt``, agora é possível treinar o modelo com o comando a seguir:
```bash
uv run train.py
``` 

Depois do modelo treinado, podemos gerar samples novas de chunk, para isso basta rodar o código:
```bash
uv run generate_samples.py
```

Agora você tem samples geradas

## Arquitetura

O modelo implementado é um VAE (Variational Autoencoder) condicional, projetado para gerar chunks 3D de Minecraft com base em condições específicas, como o bioma. A arquitetura é composta pelas seguintes partes principais:

### Encoder
O encoder é responsável por comprimir os dados de entrada (chunks e biomas) em um vetor latente de dimensão fixa. Ele utiliza:
- Camadas convolucionais 3D para extrair características espaciais.
- Embeddings para representar blocos e biomas como vetores densos.
- Camadas lineares para calcular os parâmetros da distribuição latente (média e log-variância).

![Diagrama do Encoder](media/vae_encoder.png)

### Decoder
O decoder reconstrói os chunks a partir do vetor latente e das condições fornecidas. Ele utiliza:
- Camadas deconvolucionais 3D para gerar os dados espaciais.
- Embeddings para incorporar as condições (biomas) no processo de geração.

![Diagrama do Decoder](media/vae_decoder.png)

### Pontos Fortes
- **Flexibilidade**: A inclusão de condições (como biomas) permite controlar a geração de chunks.
- **Capacidade de Generalização**: O uso de embeddings e convoluções 3D ajuda o modelo a capturar padrões espaciais complexos.
- **Regularização**: A divergência KL evita overfitting ao regularizar o espaço latente.

### Pontos Fracos
- **Custo Computacional**: Redes convolucionais 3D e o cálculo da perda podem ser computacionalmente caros.
- **Qualidade da Reconstrução**: A reconstrução pode ser limitada pela capacidade do vetor latente de capturar toda a informação relevante.
- **Treinamento Sensível**: O equilíbrio entre os termos da perda (reconstrução e KL) pode ser difícil de ajustar.