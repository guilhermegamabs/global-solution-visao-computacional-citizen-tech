import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from app.config import obterCaminhoModelo
from app.errors import ModelUnavailableError

tamanhoEntrada = 224
mediaImagenet = np.array([0.485, 0.456, 0.406], dtype=np.float32)
desvioImagenet = np.array([0.229, 0.224, 0.225], dtype=np.float32)
travaSessao = threading.Lock()
sessaoModelo = None


def modeloDisponivel():
    return Path(obterCaminhoModelo()).exists()


def obterSessao():
    global sessaoModelo
    with travaSessao:
        if sessaoModelo is None:
            caminho = Path(obterCaminhoModelo())
            if not caminho.exists():
                raise ModelUnavailableError()
            sessaoModelo = ort.InferenceSession(str(caminho), providers=["CPUExecutionProvider"])
        return sessaoModelo


def reiniciarSessao():
    global sessaoModelo
    with travaSessao:
        sessaoModelo = None


def prepararImagem(imagem):
    redimensionada = imagem.convert("RGB").resize((tamanhoEntrada, tamanhoEntrada), Image.Resampling.BILINEAR)
    matriz = np.asarray(redimensionada, dtype=np.float32) / 255.0
    return ((matriz - mediaImagenet) / desvioImagenet).transpose(2, 0, 1)


def classificarLote(imagens):
    sessao = obterSessao()
    lote = np.stack([prepararImagem(imagem) for imagem in imagens]).astype(np.float32)
    nomeEntrada = sessao.get_inputs()[0].name
    logits = sessao.run(None, {nomeEntrada: lote})[0]
    deslocados = logits - logits.max(axis=1, keepdims=True)
    exponenciais = np.exp(deslocados)
    probabilidades = exponenciais / exponenciais.sum(axis=1, keepdims=True)
    return probabilidades[:, 1].tolist()
