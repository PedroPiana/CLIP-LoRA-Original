import os, re, json
from pathlib import Path
from fnmatch import fnmatch
import matplotlib.pyplot as plt
from collections import defaultdict
import math

def coletar_acuracias(pasta_out: str, saida_json: str = "acuracias.json") -> None:
    padrao_final   = re.compile(r"Final test accuracy:\s*([0-9]+(?:\.[0-9]+)?)")
    padrao_zero    = re.compile(r"Zero-shot CLIP's test accuracy:\s*([0-9]+(?:\.[0-9]+)?)")

    resultados = {}

    for caminho in Path(pasta_out).glob("*.out"):
        with caminho.open("r", encoding="utf-8", errors="ignore") as f:
            texto = f.read()

        resultados[caminho.name] = {}

        if fnmatch(caminho.name, "CLIP-LoRA_*_1shots_*.out"):
            if (m := padrao_zero.search(texto)):
                resultados[caminho.name]["zero_shot"] = float(m.group(1))

        if (m := padrao_final.search(texto)):
            resultados[caminho.name]["final"] = float(m.group(1))

        if not resultados[caminho.name]:
            resultados.pop(caminho.name)

    Path(saida_json).parent.mkdir(parents=True, exist_ok=True)
    with open(saida_json, "w", encoding="utf-8") as fp:
        json.dump(resultados, fp, ensure_ascii=False, indent=2)

    print(f"✅ JSON salvo em {saida_json} com {len(resultados)} arquivos.")


def plotar_graficos_por_dataset(json_path: str, caminho_saida: str = "results/acuracias.png") -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        dados = json.load(f)

    # Ex: {("pigs", 1): [(1, 72.3, 42.25), (2, 85.1, None), ...] }
    dados_por_dataset_seed = defaultdict(list)
    padrao = re.compile(r"CLIP-LoRA_(.+)_(\d+)shots_(\d+)seed\.out")

    for nome_arquivo, resultados in dados.items():
        match = padrao.match(nome_arquivo)
        if not match:
            continue

        nome_dataset = match.group(1)
        num_shots = int(match.group(2))
        seed = int(match.group(3))
        final = resultados.get("final")
        zero = resultados.get("zero_shot")

        dados_por_dataset_seed[(nome_dataset, seed)].append((num_shots, final, zero))

    # Organizar quantos subplots teremos
    total_graficos = len(dados_por_dataset_seed)
    ncols = 3
    nrows = math.ceil(total_graficos / ncols)

    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axs = axs.flatten()

    for idx, ((dataset, seed), valores) in enumerate(sorted(dados_por_dataset_seed.items())):
        valores.sort(key=lambda x: x[0])
        shots = [v[0] for v in valores]
        finais = [v[1] for v in valores]
        zeros = [v[2] if v[2] is not None else None for v in valores]

        ax = axs[idx]
        ax.plot(shots, finais, marker='o', label="Final Accuracy", color="blue")
        if any(zeros):
            ax.plot(shots, zeros, marker='x', linestyle='--', label="Zero-shot Accuracy", color="orange")

        ax.set_title(f"{dataset} - Seed {seed}")
        ax.set_xlabel("Número de shots")
        ax.set_ylabel("Acurácia (%)")
        ax.grid(True)
        ax.legend()

    # Desativar eixos extras
    for i in range(idx + 1, len(axs)):
        axs[i].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    plt.savefig(caminho_saida)
    plt.close()
    print(f"📊 Gráfico geral salvo: {caminho_saida}")


# Executa as funções
coletar_acuracias("logs_scripts")
plotar_graficos_por_dataset("acuracias.json")
