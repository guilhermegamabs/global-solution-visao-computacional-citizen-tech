import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from vision import settings

azulPrimario = "#0B5FA5"
azulProfundo = "#073B6B"
vermelho = "#D32F2F"
mediaImagenet = np.array(settings.mediaImagenet, dtype=np.float32)
desvioImagenet = np.array(settings.desvioImagenet, dtype=np.float32)
limiarReferencia = 0.5
quantidadeGaleria = 8


def prepararImagem(imagem):
    redimensionada = imagem.convert("RGB").resize((settings.tamanhoEntrada, settings.tamanhoEntrada), Image.Resampling.BILINEAR)
    matriz = np.asarray(redimensionada, dtype=np.float32) / 255.0
    return ((matriz - mediaImagenet) / desvioImagenet).transpose(2, 0, 1)


def prever(sessao, imagens, tamanhoLote=64):
    nomeEntrada = sessao.get_inputs()[0].name
    probabilidades = []
    for inicio in range(0, len(imagens), tamanhoLote):
        lote = np.stack([prepararImagem(imagem) for imagem in imagens[inicio:inicio + tamanhoLote]]).astype(np.float32)
        logits = sessao.run(None, {nomeEntrada: lote})[0]
        exponenciais = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilidades.extend((exponenciais / exponenciais.sum(axis=1, keepdims=True))[:, 1].tolist())
    return np.array(probabilidades)


def calcularMetricas(reais, probabilidades, limiar):
    previstos = (probabilidades >= limiar).astype(int)
    precisao, revocacao, f1, vazio = precision_recall_fscore_support(reais, previstos, average="binary", zero_division=0)
    matriz = confusion_matrix(reais, previstos, labels=[0, 1])
    return {
        "threshold": round(float(limiar), 3),
        "precision": round(float(precisao), 4),
        "recall": round(float(revocacao), 4),
        "f1": round(float(f1), 4),
        "accuracy": round(float((previstos == reais).mean()), 4),
        "confusionMatrix": matriz.tolist(),
    }


def calcularF2(reais, probabilidades, limiar):
    metricas = calcularMetricas(reais, probabilidades, limiar)
    precisao = metricas["precision"]
    revocacao = metricas["recall"]
    if precisao + revocacao == 0:
        return 0.0
    return 5 * precisao * revocacao / (4 * precisao + revocacao)


def escolherLimiar(reais, probabilidades):
    candidatos = np.round(np.arange(0.10, 0.96, 0.05), 2)
    pontuacoes = [calcularF2(reais, probabilidades, limiar) for limiar in candidatos]
    return float(candidatos[int(np.argmax(pontuacoes))])


def varrerLimiares(reais, probabilidades):
    resultado = []
    for limiar in np.round(np.arange(0.10, 0.96, 0.05), 2):
        metricas = calcularMetricas(reais, probabilidades, limiar)
        resultado.append({chave: metricas[chave] for chave in ("threshold", "precision", "recall", "f1")})
    return resultado


def escurecer(imagem):
    return ImageEnhance.Brightness(imagem).enhance(0.45)


def clarear(imagem):
    return ImageEnhance.Brightness(imagem).enhance(1.7)


def reduzirContraste(imagem):
    return ImageEnhance.Contrast(imagem).enhance(0.5)


def desfocar(imagem):
    return imagem.filter(ImageFilter.GaussianBlur(2.5))


def adicionarRuido(imagem):
    gerador = np.random.default_rng(settings.sementeAleatoria)
    matriz = np.asarray(imagem, dtype=np.float32) + gerador.normal(0, 18, (imagem.height, imagem.width, 3))
    return Image.fromarray(np.clip(matriz, 0, 255).astype(np.uint8))


def adicionarSombra(imagem):
    matriz = np.asarray(imagem, dtype=np.float32)
    degrade = np.linspace(1.0, 0.3, imagem.width, dtype=np.float32)[None, :, None]
    return Image.fromarray(np.clip(matriz * degrade, 0, 255).astype(np.uint8))


perturbacoes = {
    "Original": None,
    "Imagem escura": escurecer,
    "Imagem clara": clarear,
    "Sombra lateral": adicionarSombra,
    "Baixo contraste": reduzirContraste,
    "Desfoque": desfocar,
    "Ruído": adicionarRuido,
}


def avaliarRobustez(sessao, imagens, reais, limiar):
    resultado = []
    for nome, funcao in perturbacoes.items():
        alteradas = imagens if funcao is None else [funcao(imagem) for imagem in imagens]
        metricas = calcularMetricas(reais, prever(sessao, alteradas), limiar)
        resultado.append({"name": nome, **{chave: metricas[chave] for chave in ("precision", "recall", "f1", "accuracy")}})
        print(f"{nome}: precisão {metricas['precision']:.3f} recall {metricas['recall']:.3f} F1 {metricas['f1']:.3f}")
    return resultado


def salvarMatriz(matriz, limiar):
    figura, eixo = plt.subplots(figsize=(4.6, 4))
    eixo.imshow(matriz, cmap="Blues")
    eixo.set_xticks([0, 1], settings.nomesClasses)
    eixo.set_yticks([0, 1], settings.nomesClasses)
    eixo.set_xlabel("Previsto")
    eixo.set_ylabel("Real")
    eixo.set_title(f"Matriz de confusão (limiar {limiar:.2f})")
    for linha in range(2):
        for coluna in range(2):
            cor = "white" if matriz[linha][coluna] > np.max(matriz) / 2 else azulProfundo
            eixo.text(coluna, linha, matriz[linha][coluna], ha="center", va="center", color=cor, fontsize=14)
    figura.tight_layout()
    figura.savefig(settings.pastaPrints / "confusionMatrix.png", dpi=160)
    plt.close(figura)


def salvarVarredura(varredura, limiar):
    figura, eixo = plt.subplots(figsize=(6, 3.8))
    limiares = [item["threshold"] for item in varredura]
    eixo.plot(limiares, [item["precision"] for item in varredura], color=azulPrimario, label="Precisão")
    eixo.plot(limiares, [item["recall"] for item in varredura], color=vermelho, label="Recall")
    eixo.plot(limiares, [item["f1"] for item in varredura], color="#1B8A5A", label="F1")
    eixo.axvline(limiar, color="gray", linestyle="--", label=f"Escolhido ({limiar:.2f})")
    eixo.set_xlabel("Limiar de confiança")
    eixo.set_ylabel("Métrica na validação")
    eixo.legend()
    eixo.grid(alpha=0.3)
    figura.tight_layout()
    figura.savefig(settings.pastaPrints / "thresholdSweep.png", dpi=160)
    plt.close(figura)


def salvarCurvasTreino():
    caminho = settings.pastaExecucoes / "history.csv"
    if not caminho.exists():
        return
    historico = pd.read_csv(caminho)
    figura, (eixoPerda, eixoF1) = plt.subplots(1, 2, figsize=(9, 3.6))
    eixoPerda.plot(historico["trainLoss"].values, color=azulPrimario, label="Treino")
    eixoPerda.plot(historico["valLoss"].values, color=vermelho, label="Validação")
    eixoPerda.set_title("Perda por época")
    eixoPerda.legend()
    eixoF1.plot(historico["valF1"].values, color="#1B8A5A")
    eixoF1.set_title("F1 de validação por época")
    for eixo in (eixoPerda, eixoF1):
        eixo.set_xlabel("Época")
        eixo.grid(alpha=0.3)
    figura.tight_layout()
    figura.savefig(settings.pastaPrints / "trainingCurves.png", dpi=160)
    plt.close(figura)


def salvarGaleriaErros(tabela, probabilidades, limiar):
    reais = tabela["label"].to_numpy()
    previstos = (probabilidades >= limiar).astype(int)
    falsosPositivos = np.where((previstos == 1) & (reais == 0))[0]
    falsosNegativos = np.where((previstos == 0) & (reais == 1))[0]
    falsosPositivos = falsosPositivos[np.argsort(-probabilidades[falsosPositivos])][:quantidadeGaleria]
    falsosNegativos = falsosNegativos[np.argsort(probabilidades[falsosNegativos])][:quantidadeGaleria]

    for nome, indices, titulo in (
        ("falsePositives", falsosPositivos, "Falsos positivos (sem fissura, previsto com fissura)"),
        ("falseNegatives", falsosNegativos, "Falsos negativos (com fissura, previsto sem fissura)"),
    ):
        if len(indices) == 0:
            continue
        figura, eixos = plt.subplots(2, 4, figsize=(10, 5.6))
        for eixo in eixos.ravel():
            eixo.axis("off")
        for eixo, indice in zip(eixos.ravel(), indices):
            eixo.imshow(Image.open(tabela.iloc[indice]["path"]).convert("RGB"))
            eixo.set_title(f"P(fissura) = {probabilidades[indice]:.2f}", fontsize=10)
        figura.suptitle(titulo)
        figura.tight_layout()
        figura.savefig(settings.pastaPrints / f"{nome}.png", dpi=140)
        plt.close(figura)
    return int(len(falsosPositivos)), int(len(falsosNegativos))


def carregarDivisao(tabela, divisao):
    recorte = tabela[tabela["split"] == divisao].reset_index(drop=True)
    imagens = [Image.open(caminho).convert("RGB") for caminho in recorte["path"]]
    return recorte, imagens, recorte["label"].to_numpy()


def executar():
    settings.pastaPrints.mkdir(parents=True, exist_ok=True)
    tabela = pd.read_csv(settings.arquivoDivisao)
    sessao = ort.InferenceSession(str(settings.caminhoOnnx), providers=["CPUExecutionProvider"])

    tabelaValidacao, imagensValidacao, reaisValidacao = carregarDivisao(tabela, "validation")
    tabelaTeste, imagensTeste, reaisTeste = carregarDivisao(tabela, "test")
    probabilidadesValidacao = prever(sessao, imagensValidacao)
    probabilidadesTeste = prever(sessao, imagensTeste)

    limiar = escolherLimiar(reaisValidacao, probabilidadesValidacao)
    varredura = varrerLimiares(reaisValidacao, probabilidadesValidacao)
    metricasTeste = calcularMetricas(reaisTeste, probabilidadesTeste, limiar)
    metricasTesteReferencia = calcularMetricas(reaisTeste, probabilidadesTeste, limiarReferencia)
    metricasValidacao = calcularMetricas(reaisValidacao, probabilidadesValidacao, limiar)

    print(f"Limiar escolhido na validação (maior F2): {limiar:.2f}")
    print(f"Teste no limiar {limiar:.2f}: {metricasTeste}")
    print(f"Teste no limiar 0.50: {metricasTesteReferencia}")

    salvarMatriz(np.array(metricasTeste["confusionMatrix"]), limiar)
    salvarVarredura(varredura, limiar)
    salvarCurvasTreino()
    falsosPositivos, falsosNegativos = salvarGaleriaErros(tabelaTeste, probabilidadesTeste, limiar)

    robustez = avaliarRobustez(sessao, imagensTeste, reaisTeste, limiar)

    resultado = {
        "model": {"architecture": "MobileNetV3-Small", "pretrained": "ImageNet", "inputSize": settings.tamanhoEntrada},
        "sample": {divisao: int((tabela["split"] == divisao).sum()) for divisao in ("train", "validation", "test")},
        "recommendedThreshold": limiar,
        "test": metricasTeste,
        "testAtDefaultThreshold": metricasTesteReferencia,
        "validation": metricasValidacao,
        "sweep": varredura,
        "robustness": robustez,
        "galleryErrors": {"falsePositives": falsosPositivos, "falseNegatives": falsosNegativos},
    }
    settings.pastaModelos.mkdir(parents=True, exist_ok=True)
    settings.caminhoMetricas.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Métricas salvas em {settings.caminhoMetricas}")


executar()
