import json
import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

segredoGerado = secrets.token_hex(32)


def obterChaveGemini():
    return os.getenv("GEMINIAPIKEY", "")


def obterModeloGemini():
    return os.getenv("GEMINIMODEL", "gemini-2.5-flash")


def obterOrigensPermitidas():
    bruto = os.getenv("ALLOWEDORIGINS", "http://localhost:8501")
    return [origem.strip() for origem in bruto.split(",") if origem.strip()]


def obterTempoLimite():
    return float(os.getenv("REQUESTTIMEOUT", "30"))


def obterUrlBanco():
    url = os.getenv("DATABASEURL", "sqlite:///orienta.db")
    for prefixo in ("postgres://", "postgresql://"):
        if url.startswith(prefixo):
            return "postgresql+psycopg://" + url[len(prefixo):]
    return url


def obterChaveBrevo():
    return os.getenv("BREVOAPIKEY", "")


def obterEmailRemetente():
    return os.getenv("SENDEREMAIL", "")


def obterNomeRemetente():
    return os.getenv("SENDERNAME", "Orienta Saúde")


def mostrarCodigoNoLog():
    return os.getenv("MOSTRARCODIGONOLOG", "").lower() == "true"


def obterSegredoApp():
    return os.getenv("SEGREDOAPP") or segredoGerado


def localizarPastaModelos():
    base = Path(__file__).resolve()
    for nivel in (1, 2):
        candidata = base.parents[nivel] / "models"
        if candidata.exists():
            return candidata
    return base.parents[1] / "models"


def obterCaminhoModelo():
    return os.getenv("MODELPATH") or str(localizarPastaModelos() / "mobilenetCrack.onnx")


def obterCaminhoMetricas():
    return os.getenv("METRICSPATH") or str(localizarPastaModelos() / "metrics.json")


def lerLimiarRecomendado():
    caminho = Path(obterCaminhoMetricas())
    if not caminho.exists():
        return None
    try:
        return float(json.loads(caminho.read_text(encoding="utf-8"))["recommendedThreshold"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def obterLimiarPadrao():
    definido = os.getenv("CRACKTHRESHOLD")
    if definido:
        return float(definido)
    recomendado = lerLimiarRecomendado()
    return recomendado if recomendado is not None else 0.5


def obterGradeBlocos():
    return int(os.getenv("TILEGRID", "3"))
