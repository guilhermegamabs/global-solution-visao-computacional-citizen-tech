import base64
import io
import json
import os
import tempfile
import time
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, UnidentifiedImageError

from app.config import obterCaminhoMetricas, obterGradeBlocos, obterLimiarPadrao
from app.errors import InvalidImageError, InvalidVideoError
from app.providers.classifierProvider import modeloDisponivel, tamanhoEntrada
from app.schemas.inspectionSchemas import (
    FramePreview,
    ImageInspection,
    ModelInfo,
    TileResult,
    TimelinePoint,
    VideoInspection,
)

ladoMaximo = 1024
quadrosMaximos = 90
passoPadrao = 5
minimoQuadrosSinalizados = 2
maximoPrevias = 4
aviso = (
    "Este resultado só ajuda a escolher onde inspecionar. A presença de fissura na imagem não indica "
    "a gravidade do dano nem a segurança da estrutura. A avaliação final é de um profissional habilitado."
)
corSinalizado = (211, 47, 47)


def abrirImagem(conteudo):
    try:
        imagem = Image.open(io.BytesIO(conteudo))
        imagem.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise InvalidImageError() from None
    imagem = imagem.convert("RGB")
    imagem.thumbnail((ladoMaximo, ladoMaximo))
    return imagem


def recortarGrade(imagem, grade):
    largura, altura = imagem.size
    recortes = []
    posicoes = []
    for linha in range(grade):
        for coluna in range(grade):
            esquerda = coluna * largura // grade
            direita = (coluna + 1) * largura // grade
            topo = linha * altura // grade
            base = (linha + 1) * altura // grade
            recortes.append(imagem.crop((esquerda, topo, direita, base)))
            posicoes.append((linha, coluna))
    return recortes, posicoes


def classificarImagem(imagem, classificar, grade, limiar):
    recortes, posicoes = recortarGrade(imagem, grade)
    probabilidades = classificar([imagem] + recortes)
    blocos = [
        TileResult(row=linha, column=coluna, probability=round(probabilidade, 4), flagged=probabilidade >= limiar)
        for (linha, coluna), probabilidade in zip(posicoes, probabilidades[1:])
    ]
    return probabilidades[0], blocos


def calcularProbabilidadeFinal(probabilidadeGeral, blocos):
    maiorBloco = max((bloco.probability for bloco in blocos), default=0.0)
    return max(probabilidadeGeral, maiorBloco)


def desenharSobreposicao(imagem, blocos, grade):
    largura, altura = imagem.size
    camada = Image.new("RGBA", imagem.size, (0, 0, 0, 0))
    pincel = ImageDraw.Draw(camada)
    for bloco in blocos:
        if not bloco.flagged:
            continue
        esquerda = bloco.column * largura // grade
        direita = (bloco.column + 1) * largura // grade
        topo = bloco.row * altura // grade
        base = (bloco.row + 1) * altura // grade
        pincel.rectangle((esquerda, topo, direita - 1, base - 1), fill=corSinalizado + (70,), outline=corSinalizado + (255,), width=4)
        pincel.text((esquerda + 8, topo + 8), f"{bloco.probability * 100:.0f}%", fill=(255, 255, 255, 255))
    return Image.alpha_composite(imagem.convert("RGBA"), camada).convert("RGB")


def codificarJpeg(imagem, qualidade=85):
    memoria = io.BytesIO()
    imagem.save(memoria, format="JPEG", quality=qualidade)
    return base64.b64encode(memoria.getvalue()).decode("ascii")


def analisarImagem(conteudo, limiar, classificar):
    limiarFinal = limiar if limiar is not None else obterLimiarPadrao()
    grade = obterGradeBlocos()
    inicio = time.perf_counter()
    imagem = abrirImagem(conteudo)
    probabilidadeGeral, blocos = classificarImagem(imagem, classificar, grade, limiarFinal)
    latencia = (time.perf_counter() - inicio) * 1000

    sinalizados = sum(1 for bloco in blocos if bloco.flagged)
    probabilidade = calcularProbabilidadeFinal(probabilidadeGeral, blocos)
    precisaInspecao = probabilidadeGeral >= limiarFinal or sinalizados > 0
    confianca = probabilidade if precisaInspecao else 1 - probabilidade

    return ImageInspection(
        label="crack" if precisaInspecao else "noCrack",
        crackProbability=round(probabilidade, 4),
        confidence=round(confianca, 4),
        threshold=limiarFinal,
        needsInspection=precisaInspecao,
        flaggedTiles=sinalizados,
        totalTiles=len(blocos),
        tiles=blocos,
        annotatedImage=codificarJpeg(desenharSobreposicao(imagem, blocos, grade)),
        latencyMs=round(latencia, 1),
        imageWidth=imagem.width,
        imageHeight=imagem.height,
        disclaimer=aviso,
    )


def gravarVideoTemporario(conteudo):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as arquivo:
        arquivo.write(conteudo)
        return arquivo.name


def manterMelhoresPrevias(previas, candidata):
    previas.append(candidata)
    previas.sort(key=lambda item: item[0], reverse=True)
    del previas[maximoPrevias:]


def analisarVideo(conteudo, limiar, classificar, passo=passoPadrao):
    limiarFinal = limiar if limiar is not None else obterLimiarPadrao()
    grade = obterGradeBlocos()
    caminho = gravarVideoTemporario(conteudo)
    captura = cv2.VideoCapture(caminho)
    try:
        if not captura.isOpened():
            raise InvalidVideoError()

        fpsOrigem = captura.get(cv2.CAP_PROP_FPS) or 30.0
        totalQuadros = int(captura.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        linhaDoTempo = []
        previas = []
        tempoInferencia = 0.0
        indice = 0

        while len(linhaDoTempo) < quadrosMaximos:
            lido, quadro = captura.read()
            if not lido:
                break
            if indice % passo == 0:
                imagem = Image.fromarray(cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB))
                imagem.thumbnail((ladoMaximo, ladoMaximo))
                inicio = time.perf_counter()
                probabilidadeGeral, blocos = classificarImagem(imagem, classificar, grade, limiarFinal)
                tempoInferencia += time.perf_counter() - inicio
                probabilidade = calcularProbabilidadeFinal(probabilidadeGeral, blocos)
                sinalizado = probabilidadeGeral >= limiarFinal or any(bloco.flagged for bloco in blocos)
                segundos = indice / fpsOrigem
                linhaDoTempo.append(
                    TimelinePoint(
                        frameIndex=indice,
                        timeSeconds=round(segundos, 2),
                        crackProbability=round(probabilidade, 4),
                        flagged=sinalizado,
                    )
                )
                if sinalizado:
                    manterMelhoresPrevias(previas, (probabilidade, indice, segundos, desenharSobreposicao(imagem, blocos, grade)))
            indice += 1
    finally:
        captura.release()
        os.unlink(caminho)

    if not linhaDoTempo:
        raise InvalidVideoError()

    analisados = len(linhaDoTempo)
    sinalizados = sum(1 for ponto in linhaDoTempo if ponto.flagged)
    duracao = totalQuadros / fpsOrigem if totalQuadros else indice / fpsOrigem

    return VideoInspection(
        needsInspection=sinalizados >= minimoQuadrosSinalizados,
        threshold=limiarFinal,
        framesAnalyzed=analisados,
        framesFlagged=sinalizados,
        flaggedRatio=round(sinalizados / analisados, 4),
        averageLatencyMs=round(tempoInferencia / analisados * 1000, 1),
        processingFps=round(analisados / tempoInferencia, 2) if tempoInferencia else 0.0,
        sourceFps=round(fpsOrigem, 2),
        durationSeconds=round(duracao, 2),
        timeline=linhaDoTempo,
        previews=[
            FramePreview(frameIndex=quadroIndice, timeSeconds=round(segundos, 2), crackProbability=round(probabilidade, 4), image=codificarJpeg(imagem))
            for probabilidade, quadroIndice, segundos, imagem in previas
        ],
        disclaimer=aviso,
    )


def lerMetricas():
    caminho = Path(obterCaminhoMetricas())
    if not caminho.exists():
        return None
    try:
        return json.loads(caminho.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def obterInformacoesModelo():
    return ModelInfo(
        name="MobileNetV3-Small (transferência de aprendizado)",
        inputSize=tamanhoEntrada,
        defaultThreshold=obterLimiarPadrao(),
        tileGrid=obterGradeBlocos(),
        available=modeloDisponivel(),
        metrics=lerMetricas(),
    )
