from pathlib import Path

import pytest

from app.dependencies import obterClassificador
from app.main import aplicacao
from app.providers.classifierProvider import modeloDisponivel

pastaExemplos = Path(__file__).resolve().parents[2] / "samples"

pytestmark = pytest.mark.skipif(
    not modeloDisponivel() or not (pastaExemplos / "imagens" / "paredeComFissura.jpg").exists(),
    reason="modelo ONNX ou exemplos não instalados",
)


def enviar(cliente, nome):
    conteudo = (pastaExemplos / "imagens" / nome).read_bytes()
    return cliente.post("/inspection/image", files={"arquivo": (nome, conteudo, "image/jpeg")}).json()


def testeParedeComFissuraEhSinalizada(cliente):
    aplicacao.dependency_overrides.pop(obterClassificador, None)
    corpo = enviar(cliente, "paredeComFissura.jpg")
    assert corpo["needsInspection"] is True
    assert corpo["flaggedTiles"] >= 1


def testeParedeSemFissuraNaoEhSinalizada(cliente):
    aplicacao.dependency_overrides.pop(obterClassificador, None)
    corpo = enviar(cliente, "paredeSemFissura.jpg")
    assert corpo["needsInspection"] is False
    assert corpo["flaggedTiles"] == 0
