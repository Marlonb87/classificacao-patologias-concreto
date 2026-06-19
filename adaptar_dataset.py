"""
ADAPTADOR DE DATASET — training/test + positive/negative
=========================================================
Converte a estrutura:
  dataset_original/
  ├── training/
  │   ├── positive/
  │   └── negative/
  └── test/
      ├── positive/
      └── negative/

Para a estrutura usada pelo pipeline.py:
  dataset/
  ├── train/
  │   ├── fissura/
  │   ├── desplacamento/
  │   ├── corrosao/
  │   └── sem_patologia/
  └── val/
      ├── fissura/
      ├── desplacamento/
      ├── corrosao/
      └── sem_patologia/

As imagens negative → copiadas automaticamente para sem_patologia/
As imagens positive → você classifica manualmente (janela OpenCV):
  f = fissura | d = desplacamento | c = corrosao | s = skip | q = salvar e sair

USO:
  python adaptar_dataset.py --origem ./caminho/do/seu/dataset

REQUISITOS:
  pip install opencv-python numpy
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

# ──────────────────────────────────────────────
# Configurações
# ──────────────────────────────────────────────
DESTINO   = Path("dataset")
EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Nomes possíveis das pastas — o script tenta cada um
NOMES_TRAIN    = ["training", "train", "Train", "Training"]
NOMES_TEST     = ["test", "val", "Test", "Val", "validation"]
NOMES_POSITIVO = ["positive", "Positive", "crack", "Crack", "defect", "anomaly"]
NOMES_NEGATIVO = ["negative", "Negative", "no_crack", "normal", "ok", "good"]


def encontrar_pasta(raiz: Path, candidatos: list[str]) -> Path | None:
    for nome in candidatos:
        p = raiz / nome
        if p.exists():
            return p
    return None


def listar_imagens(pasta: Path) -> list[Path]:
    return sorted([p for p in pasta.rglob("*") if p.suffix.lower() in EXTENSOES])


def copiar_negativos(imagens: list[Path], split: str):
    destino = DESTINO / split / "sem_patologia"
    destino.mkdir(parents=True, exist_ok=True)
    for src in imagens:
        dst = destino / src.name
        if dst.exists():
            dst = dst.with_stem(dst.stem + "_x")
        shutil.copy2(src, dst)
    print(f"  ✔ {len(imagens)} imagens → {split}/sem_patologia/")


def classificar_positivas(imagens: list[Path], split: str):
    """Abre cada imagem e aguarda tecla de classificação."""
    total = len(imagens)
    print(f"\n  Classificando {total} imagens positivas do split '{split}'...")
    print("  Teclas:  f=fissura  |  d=desplacamento  |  c=corrosao  |  s=skip  |  q=sair\n")

    contagem = {"fissura": 0, "desplacamento": 0, "corrosao": 0, "skip": 0}

    for i, caminho in enumerate(imagens):
        img = cv2.imread(str(caminho))
        if img is None:
            print(f"  [aviso] não foi possível ler: {caminho.name}")
            continue

        # Monta preview
        h, w = img.shape[:2]
        escala  = min(860 / w, 560 / h)
        preview = cv2.resize(img, (int(w * escala), int(h * escala)))

        # Barra superior com instruções
        barra = np.zeros((58, preview.shape[1], 3), dtype=np.uint8)
        cv2.putText(barra, f"[{i+1}/{total}]  {split}/positive/  {caminho.name}",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(barra, "f = fissura      d = desplacamento      c = corrosao      s = skip      q = salvar e sair",
                    (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (80, 210, 80), 1, cv2.LINE_AA)
        cv2.putText(barra,
                    f"  fissura={contagem['fissura']}  desplacamento={contagem['desplacamento']}  corrosao={contagem['corrosao']}  skip={contagem['skip']}",
                    (8, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 140), 1, cv2.LINE_AA)

        frame = np.vstack([barra, preview])
        cv2.imshow("Classificar patologia", frame)

        while True:
            tecla = cv2.waitKey(0) & 0xFF
            if tecla in (ord('f'), ord('d'), ord('c')):
                classe = {"f": "fissura", "d": "desplacamento", "c": "corrosao"}[chr(tecla)]
                pasta_dst = DESTINO / split / classe
                pasta_dst.mkdir(parents=True, exist_ok=True)
                dst = pasta_dst / caminho.name
                if dst.exists():
                    dst = dst.with_stem(dst.stem + "_x")
                shutil.copy2(caminho, dst)
                contagem[classe] += 1
                print(f"  ✔ [{i+1}/{total}] {caminho.name} → {classe}")
                break
            elif tecla == ord('s'):
                contagem["skip"] += 1
                print(f"  — [{i+1}/{total}] {caminho.name} → skipped")
                break
            elif tecla == ord('q'):
                print("\n  Progresso salvo. Encerrando classificação deste split.")
                cv2.destroyAllWindows()
                return contagem

    cv2.destroyAllWindows()
    return contagem


def resumo():
    print(f"\n{'═'*52}")
    print("  DATASET PRONTO")
    print(f"{'═'*52}")
    classes = ["fissura", "desplacamento", "corrosao", "sem_patologia"]
    print(f"  {'Classe':<18} {'Train':>7} {'Val':>7} {'Total':>7}")
    print(f"  {'-'*40}")
    total_geral = 0
    for cls in classes:
        pt = DESTINO / "train" / cls
        pv = DESTINO / "val"   / cls
        nt = len([p for p in pt.glob("*") if p.suffix.lower() in EXTENSOES]) if pt.exists() else 0
        nv = len([p for p in pv.glob("*") if p.suffix.lower() in EXTENSOES]) if pv.exists() else 0
        if nt + nv == 0:
            continue
        print(f"  {cls:<18} {nt:>7} {nv:>7} {nt+nv:>7}")
        total_geral += nt + nv
    print(f"  {'-'*40}")
    print(f"  {'TOTAL':<18} {'':>7} {'':>7} {total_geral:>7}")
    print(f"{'═'*52}")
    print(f"\n  Salvo em: {DESTINO.resolve()}")
    print("  Próximo passo: python pipeline.py --etapa tudo\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--origem", type=str, required=True,
                        help="Caminho da pasta raiz do dataset original")
    args = parser.parse_args()

    origem = Path(args.origem)
    if not origem.exists():
        print(f"\n  ERRO: pasta '{origem}' não encontrada.")
        return

    print(f"\n{'═'*52}")
    print("  ADAPTADOR DE DATASET")
    print(f"{'═'*52}")
    print(f"  Origem: {origem.resolve()}")
    print(f"  Destino: {DESTINO.resolve()}\n")

    # Detecta pastas automaticamente
    pasta_train = encontrar_pasta(origem, NOMES_TRAIN)
    pasta_val   = encontrar_pasta(origem, NOMES_TEST)

    if not pasta_train:
        print("  ERRO: pasta de treino não encontrada. Verifique --origem.")
        return
    if not pasta_val:
        print("  ERRO: pasta de teste/val não encontrada. Verifique --origem.")
        return

    print(f"  Treino detectado : {pasta_train.name}/")
    print(f"  Val detectado    : {pasta_val.name}/\n")

    # Processa cada split
    for split_nome, pasta_split in [("train", pasta_train), ("val", pasta_val)]:
        print(f"  ── Split: {split_nome} ({'─'*30})")

        pasta_neg = encontrar_pasta(pasta_split, NOMES_NEGATIVO)
        pasta_pos = encontrar_pasta(pasta_split, NOMES_POSITIVO)

        # Negativos → cópia automática
        if pasta_neg:
            imgs_neg = listar_imagens(pasta_neg)
            print(f"  Negativas: {len(imgs_neg)} imagens (cópia automática)")
            copiar_negativos(imgs_neg, split_nome)
        else:
            print(f"  [aviso] pasta negativa não encontrada em {pasta_split.name}/")

        # Positivos → classificação manual
        if pasta_pos:
            imgs_pos = listar_imagens(pasta_pos)
            print(f"  Positivas: {len(imgs_pos)} imagens (classificação manual)")
            input("  Pressione ENTER para abrir a janela de classificação...")
            contagem = classificar_positivas(imgs_pos, split_nome)
            print(f"\n  Resumo {split_nome}: {contagem}")
        else:
            print(f"  [aviso] pasta positiva não encontrada em {pasta_split.name}/")

        print()

    resumo()


if __name__ == "__main__":
    main()
