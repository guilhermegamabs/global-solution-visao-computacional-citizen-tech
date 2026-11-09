import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from app.dependencies import obterClassificador
from app.main import aplicacao


def classificadorSemFissura(imagens):
    return [0.05 for imagem in imagens]


def classificadorComUmBloco(imagens):
    probabilidades = [0.1 for imagem in imagens]
    probabilidades[2] = 0.93
    return probabilidades


def sobrescreverClassificador(funcao):
    aplicacao.dependency_overrides[obterClassificador] = lambda: funcao


def gerarImagem(largura=320, altura=240):
    matriz = np.random.default_rng(1).integers(0, 255, (altura, largura, 3), dtype=np.uint8)
    memoria = io.BytesIO()
    Image.fromarray(matriz).save(memoria, format="PNG")
    return memoria.getvalue()


def gerarVideo(quadros=20):
    caminho = Path(tempfile.mkdtemp()) / "teste.mp4"
    escritor = cv2.VideoWriter(str(caminho), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    gerador = np.random.default_rng(2)
    for posicao in range(quadros):
        escritor.write(gerador.integers(0, 255, (120, 160, 3), dtype=np.uint8))
    escritor.release()
    conteudo = caminho.read_bytes()
    caminho.unlink()
    return conteudo


def enviarImagem(cliente, conteudo, **campos):
    return cliente.post("/inspection/image", files={"arquivo": ("foto.png", conteudo, "image/png")}, data=campos)


def enviarVideo(cliente, conteudo, **campos):
    return cliente.post("/inspection/video", files={"arquivo": ("video.mp4", conteudo, "video/mp4")}, data=campos)


def testeImagemSemFissura(cliente):
    sobrescreverClassificador(classificadorSemFissura)
    resposta = enviarImagem(cliente, gerarImagem())
    corpo = resposta.json()
    assert resposta.status_code == 200
    assert corpo["needsInspection"] is False
    assert corpo["label"] == "noCrack"
    assert corpo["totalTiles"] == 9
    assert corpo["flaggedTiles"] == 0
    assert corpo["confidence"] > 0.9
    assert corpo["annotatedImage"]


def testeImagemComBlocoSinalizado(cliente):
    sobrescreverClassificador(classificadorComUmBloco)
    corpo = enviarImagem(cliente, gerarImagem()).json()
    assert corpo["needsInspection"] is True
    assert corpo["label"] == "crack"
    assert corpo["flaggedTiles"] == 1
    assert corpo["crackProbability"] == 0.93
    assert "gravidade" in corpo["disclaimer"]


def testeLimiarAlteraDecisao(cliente):
    sobrescreverClassificador(classificadorComUmBloco)
    corpo = enviarImagem(cliente, gerarImagem(), threshold="0.95").json()
    assert corpo["needsInspection"] is False
    assert corpo["threshold"] == 0.95


def testeLimiarForaDoIntervalo(cliente):
    sobrescreverClassificador(classificadorSemFissura)
    resposta = enviarImagem(cliente, gerarImagem(), threshold="1.5")
    assert resposta.status_code == 422


def testeImagemInvalida(cliente):
    sobrescreverClassificador(classificadorSemFissura)
    resposta = enviarImagem(cliente, b"isso nao e uma imagem")
    assert resposta.status_code == 422
    assert "imagem" in resposta.json()["message"]


def testeImagemGrandeDemais(cliente):
    sobrescreverClassificador(classificadorSemFissura)
    resposta = enviarImagem(cliente, b"0" * (10 * 1024 * 1024 + 10))
    assert resposta.status_code == 413


def testeExigeLogin(clienteAnonimo):
    sobrescreverClassificador(classificadorSemFissura)
    assert enviarImagem(clienteAnonimo, gerarImagem()).status_code == 401
    assert enviarVideo(clienteAnonimo, gerarVideo(5)).status_code == 401


def testeVideoSemFissura(cliente):
    sobrescreverClassificador(classificadorSemFissura)
    resposta = enviarVideo(cliente, gerarVideo())
    corpo = resposta.json()
    assert resposta.status_code == 200
    assert corpo["framesAnalyzed"] == 4
    assert corpo["framesFlagged"] == 0
    assert corpo["needsInspection"] is False
    assert corpo["processingFps"] > 0
    assert corpo["previews"] == []
    assert len(corpo["timeline"]) == 4


def testeVideoComFissura(cliente):
    sobrescreverClassificador(classificadorComUmBloco)
    corpo = enviarVideo(cliente, gerarVideo()).json()
    assert corpo["framesFlagged"] == 4
    assert corpo["flaggedRatio"] == 1.0
    assert corpo["needsInspection"] is True
    assert 1 <= len(corpo["previews"]) <= 4
    assert corpo["averageLatencyMs"] >= 0


def testeVideoInvalido(cliente):
    sobrescreverClassificador(classificadorSemFissura)
    resposta = enviarVideo(cliente, b"nao e video")
    assert resposta.status_code == 422


def testeInformacoesDoModelo(cliente):
    corpo = cliente.get("/inspection/model").json()
    assert corpo["inputSize"] == 224
    assert corpo["tileGrid"] == 3
    assert "available" in corpo
