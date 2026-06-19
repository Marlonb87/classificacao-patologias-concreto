"""
PIPELINE COMPLETO — Diagnóstico de Patologias em Concreto Armado
=================================================================
Executa em sequência:
  1. Organização do dataset (classificação manual das imagens)
  2. Pré-processamento e análise exploratória (EDA)
  3. Treinamento da CNN (ResNet-18 com transfer learning)
  4. Avaliação completa com métricas e visualizações

ESTRUTURA GERADA:
  projeto_patologias/
  ├── pipeline.py              ← este arquivo
  ├── dataset/
  │   ├── train/  fissura / desplacamento / corrosao
  │   └── val/    fissura / desplacamento / corrosao
  ├── modelos/
  │   └── melhor_modelo.pth
  └── resultados/
      ├── eda/
      │   ├── distribuicao_classes.png
      │   ├── amostras_por_classe.png
      │   └── dimensoes_originais.png
      ├── treinamento/
      │   └── historico_treinamento.png
      └── avaliacao/
          ├── matriz_confusao.png
          ├── metricas_por_classe.png
          ├── exemplos_predicoes.png
          └── relatorio_metricas.txt

USO:
  # Etapa 1 — organizar imagens manualmente (abre janela OpenCV)
  python pipeline.py --etapa organizar --pasta_imagens /caminho/imagens

  # Etapa 2 — pré-processamento e EDA
  python pipeline.py --etapa eda

  # Etapa 3 — treinamento
  python pipeline.py --etapa treinar --epocas 15

  # Etapa 4 — avaliação
  python pipeline.py --etapa avaliar

  # Rodar tudo de uma vez (sem etapa 1, que é interativa)
  python pipeline.py --etapa tudo

REQUISITOS:
  pip install torch torchvision opencv-python matplotlib numpy scikit-learn
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import os
import sys
import copy
import time
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, accuracy_score
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÕES GLOBAIS
# ══════════════════════════════════════════════════════════════════════════════
CLASSES         = ["fissura", "desplacamento", "corrosao"]
CORES_CLASSES   = {"fissura": "#E74C3C", "desplacamento": "#3498DB", "corrosao": "#2ECC71"}
IMG_SIZE        = 224
BATCH_SIZE      = 32
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-4
VAL_SPLIT       = 0.2
SEED            = 42
EXTENSOES       = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
TAMANHO_PREVIEW = (800, 600)

BASE_DIR    = Path("dataset")
MODELO_PATH = Path("modelos/melhor_modelo.pth")
RES_EDA     = Path("resultados/eda")
RES_TRAIN   = Path("resultados/treinamento")
RES_AVAL    = Path("resultados/avaliacao")

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITÁRIOS
# ══════════════════════════════════════════════════════════════════════════════
def titulo(texto: str):
    print(f"\n{'═'*60}")
    print(f"  {texto}")
    print(f"{'═'*60}")


def ok(texto: str):
    print(f"  ✔  {texto}")


def info(texto: str):
    print(f"  →  {texto}")


def erro(texto: str):
    print(f"  ✘  {texto}")
    sys.exit(1)


def listar_imagens(pasta: Path) -> list[Path]:
    return sorted([p for p in pasta.rglob("*") if p.suffix.lower() in EXTENSOES])


def listar_por_classe(split: str) -> dict[str, list[Path]]:
    resultado = {}
    for cls in CLASSES:
        pasta = BASE_DIR / split / cls
        if pasta.exists():
            imgs = [p for p in pasta.glob("*") if p.suffix.lower() in EXTENSOES]
            resultado[cls] = imgs
    return resultado


def carregar_imagem_rgb(caminho: Path, size=(IMG_SIZE, IMG_SIZE)) -> np.ndarray | None:
    img = cv2.imread(str(caminho))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, size)


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 1 — ORGANIZAÇÃO DO DATASET
# ══════════════════════════════════════════════════════════════════════════════
def criar_estrutura_dataset():
    for split in ["train", "val"]:
        for cls in CLASSES:
            (BASE_DIR / split / cls).mkdir(parents=True, exist_ok=True)


# classificação manual removida — dataset já vem organizado por pastas


def distribuir_dataset(rotulos: dict[Path, str]) -> tuple[int, dict]:
    por_classe: dict[str, list[Path]] = {c: [] for c in CLASSES}
    for caminho, classe in rotulos.items():
        por_classe[classe].append(caminho)

    random.shuffle_seed = SEED
    copiados = 0
    for classe, arquivos in por_classe.items():
        random.shuffle(arquivos)
        n_val = max(1, int(len(arquivos) * VAL_SPLIT))
        splits = {"val": arquivos[:n_val], "train": arquivos[n_val:]}
        for split, lista in splits.items():
            for src in lista:
                dst = BASE_DIR / split / classe / src.name
                if dst.exists():
                    dst = dst.with_stem(dst.stem + "_dup")
                shutil.copy2(src, dst)
                copiados += 1

    contagem = {c: len(v) for c, v in por_classe.items()}
    return copiados, contagem


def etapa_organizar(pasta_imagens: str):
    """
    Lê automaticamente as subpastas de pasta_imagens como classes
    e divide em train/val sem nenhuma interação manual.

    Estrutura esperada:
      pasta_imagens/
      ├── fissura/
      ├── desplacamento/
      └── corrosao/
    """
    titulo("ETAPA 1 — Organização do Dataset (automática)")
    pasta = Path(pasta_imagens)
    if not pasta.exists():
        erro(f"Pasta '{pasta}' não encontrada.")

    # Detecta subpastas como classes
    subpastas = sorted([p for p in pasta.iterdir() if p.is_dir()])
    if not subpastas:
        erro(f"Nenhuma subpasta encontrada em '{pasta}'.")

    info(f"Classes detectadas: {[p.name for p in subpastas]}")
    criar_estrutura_dataset()

    contagem_total = 0
    random.seed(SEED)

    print(f"\n  {'Classe':<18} {'Total':>7} {'Train':>7} {'Val':>7}")
    print(f"  {'-'*46}")

    for subpasta in subpastas:
        classe = subpasta.name

        # Aceita qualquer classe que bata com CLASSES (case-insensitive)
        classe_norm = next(
            (c for c in CLASSES if c.lower() == classe.lower()), classe
        )

        imagens = listar_imagens(subpasta)
        if not imagens:
            print(f"  [aviso] Nenhuma imagem em '{subpasta.name}/', pulando.")
            continue

        random.shuffle(imagens)
        n_val   = max(1, int(len(imagens) * VAL_SPLIT))
        n_train = len(imagens) - n_val
        splits  = {"val": imagens[:n_val], "train": imagens[n_val:]}

        for split, lista in splits.items():
            destino = BASE_DIR / split / classe_norm
            destino.mkdir(parents=True, exist_ok=True)
            for src in lista:
                dst = destino / src.name
                if dst.exists():
                    dst = dst.with_stem(dst.stem + "_dup")
                shutil.copy2(src, dst)

        contagem_total += len(imagens)
        print(f"  {classe_norm:<18} {len(imagens):>7} {n_train:>7} {n_val:>7}")

    print(f"  {'-'*46}")
    print(f"  {'TOTAL':<18} {contagem_total:>7}")
    ok(f"Dataset organizado em: {BASE_DIR.resolve()}")
    info("Próximo passo: --etapa eda")


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 2 — PRÉ-PROCESSAMENTO E EDA
# ══════════════════════════════════════════════════════════════════════════════
def analisar_dimensoes(dataset: dict[str, list[Path]]) -> dict:
    stats = {}
    for cls, arquivos in dataset.items():
        dims = []
        for p in arquivos:
            img = cv2.imread(str(p))
            if img is not None:
                dims.append((img.shape[1], img.shape[0]))
        if dims:
            ws, hs = zip(*dims)
            stats[cls] = {
                "total": len(arquivos),
                "w_med": int(np.median(ws)), "h_med": int(np.median(hs)),
                "w_min": min(ws), "w_max": max(ws),
            }
        else:
            stats[cls] = {"total": 0}
    return stats


def plotar_distribuicao(stats_train, stats_val):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Distribuição de Classes no Dataset", fontsize=13, fontweight="bold")
    for ax, (nome, stats) in zip(axes, [("Treino", stats_train), ("Validação", stats_val)]):
        classes = list(stats.keys())
        totais  = [stats[c].get("total", 0) for c in classes]
        cores   = [CORES_CLASSES.get(c, "#888") for c in classes]
        bars = ax.bar(classes, totais, color=cores, edgecolor="white", linewidth=1.2)
        ax.set_title(f"Split: {nome}")
        ax.set_ylabel("Quantidade de imagens")
        for bar, val in zip(bars, totais):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylim(0, max(totais + [1]) * 1.15)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RES_EDA / "distribuicao_classes.png", dpi=150, bbox_inches="tight")
    plt.close()
    ok("distribuicao_classes.png")


def plotar_amostras(dataset: dict[str, list[Path]], n_por_classe=3):
    fig, axes = plt.subplots(len(CLASSES), n_por_classe,
                             figsize=(n_por_classe * 4, len(CLASSES) * 4))
    fig.suptitle("Amostras por Classe (após redimensionamento)", fontsize=12, fontweight="bold")
    for row, cls in enumerate(CLASSES):
        arquivos = dataset.get(cls, [])
        amostras = random.sample(arquivos, min(n_por_classe, len(arquivos)))
        for col in range(n_por_classe):
            ax = axes[row][col] if len(CLASSES) > 1 else axes[col]
            if col < len(amostras):
                img = carregar_imagem_rgb(amostras[col])
                if img is not None:
                    ax.imshow(img)
                    ax.set_title(f"{cls}\n{amostras[col].name[:20]}", fontsize=8)
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(RES_EDA / "amostras_por_classe.png", dpi=150, bbox_inches="tight")
    plt.close()
    ok("amostras_por_classe.png")


def plotar_dimensoes(stats: dict):
    classes = [c for c in stats if stats[c].get("w_med")]
    if not classes:
        return
    w_meds = [stats[c]["w_med"] for c in classes]
    h_meds = [stats[c]["h_med"] for c in classes]
    x, width = np.arange(len(classes)), 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, w_meds, width, label="Largura mediana", color="#3498DB")
    ax.bar(x + width/2, h_meds, width, label="Altura mediana",  color="#E74C3C")
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylabel("Pixels")
    ax.set_title("Dimensões medianas das imagens originais por classe", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RES_EDA / "dimensoes_originais.png", dpi=150, bbox_inches="tight")
    plt.close()
    ok("dimensoes_originais.png")


def augmentar_imagem(img: np.ndarray) -> list[np.ndarray]:
    return [
        cv2.flip(img, 1),
        cv2.flip(img, 0),
        cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
        np.clip(img.astype(np.int16) + 30, 0, 255).astype(np.uint8),
    ]


def balancear_classes(dataset: dict[str, list[Path]]) -> dict:
    totais   = {c: len(v) for c, v in dataset.items()}
    max_total = max(totais.values())
    geradas  = defaultdict(int)

    for cls, arquivos in dataset.items():
        faltam = max_total - len(arquivos)
        if faltam <= 0:
            continue
        info(f"Augmentando '{cls}': {len(arquivos)} → {max_total} (+{faltam})")
        pasta_cls = arquivos[0].parent
        pool = arquivos.copy(); random.shuffle(pool)
        idx = geradas_cls = 0
        while geradas_cls < faltam:
            src = pool[idx % len(pool)]
            img = cv2.imread(str(src))
            if img is None:
                idx += 1; continue
            img_r = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            for aug_img in augmentar_imagem(img_r):
                if geradas_cls >= faltam:
                    break
                nome = pasta_cls / f"aug_{idx:04d}_{geradas_cls:04d}{src.suffix}"
                cv2.imwrite(str(nome), aug_img)
                geradas_cls += 1
            idx += 1
        geradas[cls] = geradas_cls
    return dict(geradas)


def etapa_eda():
    titulo("ETAPA 2 — Pré-processamento e EDA")
    RES_EDA.mkdir(parents=True, exist_ok=True)

    train_data = listar_por_classe("train")
    val_data   = listar_por_classe("val")

    if not train_data:
        erro("Dataset não encontrado. Execute primeiro: --etapa organizar")

    info("Analisando dimensões das imagens...")
    stats_train = analisar_dimensoes(train_data)
    stats_val   = analisar_dimensoes(val_data)

    info("Gerando gráficos EDA...")
    plotar_distribuicao(stats_train, stats_val)
    plotar_amostras(train_data)
    plotar_dimensoes(stats_train)

    info("Balanceando classes com data augmentation...")
    geradas = balancear_classes(train_data)

    print(f"\n  {'Classe':<16} {'Train':>6} {'Val':>6} {'Aug':>6} {'W_med':>6} {'H_med':>6}")
    print(f"  {'-'*48}")
    for cls in CLASSES:
        t = stats_train.get(cls, {}).get("total", 0)
        v = stats_val.get(cls, {}).get("total", 0)
        a = geradas.get(cls, 0)
        w = stats_train.get(cls, {}).get("w_med", "-")
        h = stats_train.get(cls, {}).get("h_med", "-")
        print(f"  {cls:<16} {t:>6} {v:>6} {a:>6} {str(w):>6} {str(h):>6}")

    ok(f"Gráficos salvos em: {RES_EDA.resolve()}")


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 3 — TREINAMENTO
# ══════════════════════════════════════════════════════════════════════════════
def get_transforms():
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf


def criar_modelo(n_classes: int, device: torch.device) -> nn.Module:
    modelo = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in modelo.parameters():
        param.requires_grad = False
    for param in modelo.layer4.parameters():
        param.requires_grad = True
    n_features = modelo.fc.in_features
    modelo.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(n_features, n_classes))
    return modelo.to(device)


def treinar_loop(modelo, dataloaders, criterion, optimizer, scheduler,
                 n_epocas, device) -> tuple[nn.Module, dict]:
    historico = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    melhor_wts = copy.deepcopy(modelo.state_dict())
    melhor_acc = 0.0
    inicio = time.time()

    for epoca in range(n_epocas):
        print(f"\n  Época {epoca+1}/{n_epocas}  {'─'*35}")
        for fase in ["train", "val"]:
            modelo.train() if fase == "train" else modelo.eval()
            loss_total = acertos = 0
            for inputs, labels in dataloaders[fase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(fase == "train"):
                    outputs = modelo(inputs)
                    loss    = criterion(outputs, labels)
                    preds   = outputs.argmax(dim=1)
                    if fase == "train":
                        loss.backward(); optimizer.step()
                loss_total += loss.item() * inputs.size(0)
                acertos    += (preds == labels).sum().item()

            n = len(dataloaders[fase].dataset)
            epoch_loss = loss_total / n
            epoch_acc  = acertos / n
            historico[f"{'train' if fase == 'train' else 'val'}_loss"].append(epoch_loss)
            historico[f"{'train' if fase == 'train' else 'val'}_acc"].append(epoch_acc)
            print(f"    {fase:<6}  loss={epoch_loss:.4f}  acc={epoch_acc:.4f}")

            if fase == "val" and epoch_acc > melhor_acc:
                melhor_acc = epoch_acc
                melhor_wts = copy.deepcopy(modelo.state_dict())

        if scheduler:
            scheduler.step()

    tempo = time.time() - inicio
    print(f"\n  Concluído em {tempo/60:.1f} min  |  Melhor val_acc: {melhor_acc:.4f}")
    modelo.load_state_dict(melhor_wts)
    return modelo, historico


def plotar_historico(historico: dict):
    epocas = range(1, len(historico["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Histórico de Treinamento — CNN ResNet-18", fontsize=13, fontweight="bold")

    ax1.plot(epocas, historico["train_loss"], "o-", label="Treino",    color="#E74C3C")
    ax1.plot(epocas, historico["val_loss"],  "s--", label="Validação", color="#3498DB")
    ax1.set_title("Loss por Época"); ax1.set_xlabel("Época"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epocas, [a*100 for a in historico["train_acc"]], "o-", label="Treino",    color="#E74C3C")
    ax2.plot(epocas, [a*100 for a in historico["val_acc"]],  "s--", label="Validação", color="#3498DB")
    ax2.set_title("Acurácia por Época"); ax2.set_xlabel("Época"); ax2.set_ylabel("Acurácia (%)")
    ax2.set_ylim(0, 100); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RES_TRAIN / "historico_treinamento.png", dpi=150, bbox_inches="tight")
    plt.close()
    ok("historico_treinamento.png")


def etapa_treinar(n_epocas: int, batch: int, lr: float):
    titulo("ETAPA 3 — Treinamento da CNN")
    RES_TRAIN.mkdir(parents=True, exist_ok=True)
    Path("modelos").mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info(f"Dispositivo: {device}")
    if device.type == "cuda":
        info(f"GPU: {torch.cuda.get_device_name(0)}")

    train_tf, val_tf = get_transforms()
    image_datasets = {
        "train": datasets.ImageFolder(str(BASE_DIR / "train"), transform=train_tf),
        "val":   datasets.ImageFolder(str(BASE_DIR / "val"),   transform=val_tf),
    }
    dataloaders = {
        s: DataLoader(ds, batch_size=batch, shuffle=(s == "train"), num_workers=0)
        for s, ds in image_datasets.items()
    }

    info(f"Classes: {image_datasets['train'].classes}")
    info(f"Treino: {len(image_datasets['train'])} imagens | Val: {len(image_datasets['val'])} imagens")

    contagem = torch.tensor(
        [len(list((BASE_DIR / "train" / c).glob("*"))) for c in image_datasets["train"].classes],
        dtype=torch.float
    )
    pesos = (1.0 / contagem); pesos = (pesos / pesos.sum()).to(device)
    criterion = nn.CrossEntropyLoss(weight=pesos)

    modelo    = criar_modelo(len(CLASSES), device)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, modelo.parameters()),
        lr=lr, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epocas)

    modelo, historico = treinar_loop(modelo, dataloaders, criterion, optimizer,
                                     scheduler, n_epocas, device)

    torch.save(modelo.state_dict(), MODELO_PATH)
    ok(f"Modelo salvo em: {MODELO_PATH}")
    plotar_historico(historico)


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 4 — AVALIAÇÃO
# ══════════════════════════════════════════════════════════════════════════════
def carregar_modelo_treinado(n_classes: int, device: torch.device) -> nn.Module:
    modelo = models.resnet18(weights=None)
    n_features = modelo.fc.in_features
    modelo.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(n_features, n_classes))
    modelo.load_state_dict(torch.load(MODELO_PATH, map_location=device))
    modelo.to(device).eval()
    return modelo


def inferir(modelo, dataloader, device):
    softmax = nn.Softmax(dim=1)
    preds_all, labels_all, probs_all, imgs_all = [], [], [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = modelo(inputs)
            preds_all.extend(outputs.argmax(dim=1).cpu().numpy())
            labels_all.extend(labels.numpy())
            probs_all.extend(softmax(outputs).cpu().numpy())
            imgs_all.extend(inputs.cpu())
    return np.array(labels_all), np.array(preds_all), np.array(probs_all), imgs_all


def plotar_matriz_confusao(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Matriz de Confusão", fontsize=13, fontweight="bold")
    for ax, (data, titulo_ax, fmt) in zip(axes, [
        (cm,      "Contagem Absoluta", "d"),
        (cm_norm, "Normalizada",      ".2%"),
    ]):
        im = ax.imshow(data, interpolation="nearest",
                       cmap=plt.cm.Blues if fmt == "d" else plt.cm.RdYlGn)
        plt.colorbar(im, ax=ax)
        ticks = np.arange(len(classes))
        ax.set_xticks(ticks); ax.set_xticklabels(classes, rotation=30, ha="right")
        ax.set_yticks(ticks); ax.set_yticklabels(classes)
        thresh = data.max() / 2.0
        for i in range(len(classes)):
            for j in range(len(classes)):
                cor = "white" if data[i, j] > thresh else "black"
                ax.text(j, i, f"{data[i,j]:{fmt}}", ha="center", va="center",
                        fontsize=11, color=cor, fontweight="bold")
        ax.set_ylabel("Real"); ax.set_xlabel("Predito"); ax.set_title(titulo_ax)
    plt.tight_layout()
    plt.savefig(RES_AVAL / "matriz_confusao.png", dpi=150, bbox_inches="tight")
    plt.close()
    ok("matriz_confusao.png")


def plotar_metricas_por_classe(y_true, y_pred, classes):
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(len(classes)))
    x, w = np.arange(len(classes)), 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, prec, w, label="Precision", color="#3498DB")
    ax.bar(x,     rec,  w, label="Recall",    color="#E74C3C")
    ax.bar(x + w, f1,   w, label="F1-Score",  color="#2ECC71")
    ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Score")
    ax.set_title("Precision, Recall e F1-Score por Classe", fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for bar in ax.patches:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(RES_AVAL / "metricas_por_classe.png", dpi=150, bbox_inches="tight")
    plt.close()
    ok("metricas_por_classe.png")


def plotar_exemplos(imgs, y_true, y_pred, probs, classes, n=12):
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    def desnorm(t):
        img = t.permute(1, 2, 0).numpy()
        return np.clip(img * std + mean, 0, 1)

    corretos   = [i for i in range(len(y_true)) if y_true[i] == y_pred[i]]
    incorretos = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]
    amostras   = (random.sample(corretos,   min(n//2, len(corretos)))   +
                  random.sample(incorretos, min(n//2, len(incorretos))))
    if not amostras:
        return

    cols = 6
    rows = (len(amostras) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    fig.suptitle("Exemplos de Predições  (✔ correto | ✘ incorreto)", fontsize=12, fontweight="bold")
    axes = axes.flatten()

    for ax_idx, img_idx in enumerate(amostras):
        ok_pred = y_true[img_idx] == y_pred[img_idx]
        cor = "#27AE60" if ok_pred else "#E74C3C"
        axes[ax_idx].imshow(desnorm(imgs[img_idx]))
        axes[ax_idx].set_title(
            f"{'✔' if ok_pred else '✘'} Real: {classes[y_true[img_idx]]}\n"
            f"Pred: {classes[y_pred[img_idx]]} ({probs[img_idx][y_pred[img_idx]]*100:.1f}%)",
            fontsize=8, color=cor, fontweight="bold"
        )
        for spine in axes[ax_idx].spines.values():
            spine.set_edgecolor(cor); spine.set_linewidth(2.5)
        axes[ax_idx].set_xticks([]); axes[ax_idx].set_yticks([])

    for ax in axes[len(amostras):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(RES_AVAL / "exemplos_predicoes.png", dpi=150, bbox_inches="tight")
    plt.close()
    ok("exemplos_predicoes.png")


def salvar_relatorio(y_true, y_pred, classes, acc):
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    cm = confusion_matrix(y_true, y_pred)
    path = RES_AVAL / "relatorio_metricas.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("RELATÓRIO DE AVALIAÇÃO — CNN ResNet-18\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Acurácia Global: {acc*100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n\n")
        f.write("Matriz de Confusão (linhas=real, colunas=predito):\n")
        f.write("\t".join([""] + classes) + "\n")
        for cls, row in zip(classes, cm):
            f.write(cls + "\t" + "\t".join(map(str, row)) + "\n")
    ok("relatorio_metricas.txt")


def etapa_avaliar():
    titulo("ETAPA 4 — Avaliação do Modelo")
    RES_AVAL.mkdir(parents=True, exist_ok=True)

    if not MODELO_PATH.exists():
        erro(f"Modelo não encontrado em '{MODELO_PATH}'. Execute primeiro --etapa treinar")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_ds = datasets.ImageFolder(str(BASE_DIR / "val"), transform=val_tf)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    classes_ds = val_ds.classes

    info(f"Classes: {classes_ds}  |  Imagens: {len(val_ds)}")
    modelo = carregar_modelo_treinado(len(classes_ds), device)

    info("Realizando inferência...")
    y_true, y_pred, probs, imgs = inferir(modelo, val_loader, device)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n  Acurácia Global: {acc*100:.2f}%\n")
    print(classification_report(y_true, y_pred, target_names=classes_ds, digits=4))

    info("Gerando visualizações...")
    plotar_matriz_confusao(y_true, y_pred, classes_ds)
    plotar_metricas_por_classe(y_true, y_pred, classes_ds)
    plotar_exemplos(imgs, y_true, y_pred, probs, classes_ds)
    salvar_relatorio(y_true, y_pred, classes_ds, acc)

    ok(f"Resultados salvos em: {RES_AVAL.resolve()}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de Diagnóstico de Patologias em Concreto Armado",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--etapa", type=str, required=True,
                        choices=["organizar", "eda", "treinar", "avaliar", "tudo"],
                        help=(
                            "organizar  — classifica manualmente as imagens\n"
                            "eda        — pré-processamento e análise exploratória\n"
                            "treinar    — treina a CNN ResNet-18\n"
                            "avaliar    — gera métricas e visualizações\n"
                            "tudo       — executa eda + treinar + avaliar"
                        ))
    parser.add_argument("--pasta_imagens", type=str, default="",
                        help="Pasta com imagens brutas (obrigatório para --etapa organizar)")
    parser.add_argument("--epocas", type=int,   default=15)
    parser.add_argument("--batch",  type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",     type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    if args.etapa == "organizar":
        if not args.pasta_imagens:
            erro("Informe --pasta_imagens para a etapa 'organizar'.")
        etapa_organizar(args.pasta_imagens)

    elif args.etapa == "eda":
        etapa_eda()

    elif args.etapa == "treinar":
        etapa_treinar(args.epocas, args.batch, args.lr)

    elif args.etapa == "avaliar":
        etapa_avaliar()

    elif args.etapa == "tudo":
        etapa_eda()
        etapa_treinar(args.epocas, args.batch, args.lr)
        etapa_avaliar()

    print("\n  Pipeline finalizado!\n")


if __name__ == "__main__":
    main()
