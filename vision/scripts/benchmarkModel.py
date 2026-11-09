import json
import os
import platform
import statistics
import subprocess
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

from vision import settings

execucoesUnitarias = 200
execucoesFoto = 100
aquecimento = 20
largura = 1024
altura = 768
grade = 3


def descobrirProcessador():
    try:
        saida = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        nome = saida.stdout.strip()
        if nome:
            return nome
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "desconhecido"


def prepararImagem(imagem):
    redimensionada = imagem.resize((settings.tamanhoEntrada, settings.tamanhoEntrada), Image.Resampling.BILINEAR)
    matriz = np.asarray(redimensionada, dtype=np.float32) / 255.0
    media = np.array(settings.mediaImagenet, dtype=np.float32)
    desvio = np.array(settings.desvioImagenet, dtype=np.float32)
    return ((matriz - media) / desvio).transpose(2, 0, 1)


def recortarFoto(imagem):
    recortes = [imagem]
    for linha in range(grade):
        for coluna in range(grade):
            recortes.append(
                imagem.crop((coluna * largura // grade, linha * altura // grade, (coluna + 1) * largura // grade, (linha + 1) * altura // grade))
            )
    return recortes


def medir(funcao, execucoes, descartadas):
    for repeticao in range(descartadas):
        funcao()
    tempos = []
    for repeticao in range(execucoes):
        inicio = time.perf_counter()
        funcao()
        tempos.append((time.perf_counter() - inicio) * 1000)
    tempos.sort()
    media = statistics.fmean(tempos)
    return {
        "meanMs": round(media, 2),
        "medianMs": round(statistics.median(tempos), 2),
        "p95Ms": round(tempos[int(len(tempos) * 0.95) - 1], 2),
        "perSecond": round(1000 / media, 2),
        "runs": execucoes,
    }


def executar():
    sessao = ort.InferenceSession(str(settings.caminhoOnnx), providers=["CPUExecutionProvider"])
    nomeEntrada = sessao.get_inputs()[0].name
    gerador = np.random.default_rng(settings.sementeAleatoria)
    unitario = gerador.standard_normal((1, 3, settings.tamanhoEntrada, settings.tamanhoEntrada)).astype(np.float32)
    lote = gerador.standard_normal((grade * grade + 1, 3, settings.tamanhoEntrada, settings.tamanhoEntrada)).astype(np.float32)
    foto = Image.fromarray(gerador.integers(0, 255, (altura, largura, 3), dtype=np.uint8))

    resultadoUnitario = medir(lambda: sessao.run(None, {nomeEntrada: unitario}), execucoesUnitarias, aquecimento)
    resultadoLote = medir(lambda: sessao.run(None, {nomeEntrada: lote}), execucoesFoto, 5)

    def analisarFotoCompleta():
        recortes = recortarFoto(foto)
        entrada = np.stack([prepararImagem(recorte) for recorte in recortes]).astype(np.float32)
        sessao.run(None, {nomeEntrada: entrada})

    resultadoFoto = medir(analisarFotoCompleta, execucoesFoto, 5)

    relatorio = {
        "equipment": {
            "processor": descobrirProcessador(),
            "logicalCores": os.cpu_count(),
            "system": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "onnxruntime": ort.__version__,
            "providers": sessao.get_providers(),
        },
        "singleImage224": {"resolution": f"{settings.tamanhoEntrada}x{settings.tamanhoEntrada}", **resultadoUnitario},
        "tileBatch": {"resolution": f"{settings.tamanhoEntrada}x{settings.tamanhoEntrada}", "batchSize": grade * grade + 1, **resultadoLote},
        "fullPhotoPipeline": {
            "resolution": f"{largura}x{altura}",
            "description": "recorte em grade 3x3, pré-processamento e inferência de 10 imagens",
            **resultadoFoto,
        },
    }
    settings.pastaRelatorio.mkdir(parents=True, exist_ok=True)
    settings.caminhoBenchmark.write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(relatorio, ensure_ascii=False, indent=2))


executar()
