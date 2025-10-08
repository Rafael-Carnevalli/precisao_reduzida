import random
import sys

# --- Parâmetros ---
# Número de pontos a serem gerados
N = 10_000_000
# Número de clusters
K = 500

# Definições dos clusters: geradas programaticamente
# Vamos criar K clusters distribuídos ao longo de um grande intervalo
cluster_defs = []
for i in range(K):
    media = i * 100  # Espalha as médias dos clusters
    desvio_padrao = random.uniform(10, 30)
    cluster_defs.append((media, desvio_padrao))


if K != len(cluster_defs):
    print(f"Erro: K ({K}) não corresponde ao número de definições de cluster ({len(cluster_defs)})")
    sys.exit(1)

# --- Geração do arquivo de dados ---
print(f"Gerando {N:,} pontos de dados para o arquivo 'dados.csv'...")
try:
    with open('dados.csv', 'w') as f:
        for i in range(N):
            # Escolhe um dos clusters aleatoriamente para gerar um ponto
            cluster = random.choice(cluster_defs)
            media, desvio_padrao = cluster
            
            # Gera um ponto a partir de uma distribuição normal (gaussiana)
            ponto = random.normalvariate(media, desvio_padrao)
            f.write(f"{ponto:.6f}\n")
except IOError as e:
    print(f"Erro ao escrever em dados.csv: {e}")
    sys.exit(1)

# --- Geração do arquivo de centróides iniciais ---
# Vamos usar as médias dos clusters como ponto de partida, com uma pequena perturbação
print(f"Gerando {K} centróides iniciais para o arquivo 'centroides_iniciais.csv'...")
try:
    with open('centroides_iniciais.csv', 'w') as f:
        for media, _ in cluster_defs:
            # Adiciona um pequeno ruído para não começar com a resposta perfeita
            centroide_inicial = media + random.uniform(-10, 10)
            f.write(f"{centroide_inicial:.6f}\n")
except IOError as e:
    print(f"Erro ao escrever em centroides_iniciais.csv: {e}")
    sys.exit(1)

print("\nArquivos 'dados.csv' e 'centroides_iniciais.csv' gerados com sucesso!")
print("Agora você pode executar o programa k-means com esses arquivos.")
