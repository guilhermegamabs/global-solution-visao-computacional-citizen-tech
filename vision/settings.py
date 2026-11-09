import os
from pathlib import Path

raizProjeto = Path(__file__).resolve().parent.parent
pastaVisao = raizProjeto / "vision"
pastaDados = Path(os.getenv("VISIONDATA") or pastaVisao / "data")
pastaBruta = pastaDados / "raw"
arquivoDivisao = pastaDados / "splits.csv"
pastaExecucoes = pastaVisao / "runs"
pastaModelos = raizProjeto / "models"
pastaPrints = raizProjeto / "docs" / "prints"
pastaRelatorio = raizProjeto / "docs" / "relatorio"
caminhoPesos = pastaExecucoes / "best.pt"
caminhoOnnx = pastaModelos / "mobilenetCrack.onnx"
caminhoMetricas = pastaModelos / "metrics.json"
caminhoBenchmark = pastaRelatorio / "benchmark.json"

tamanhoEntrada = 224
mediaImagenet = (0.485, 0.456, 0.406)
desvioImagenet = (0.229, 0.224, 0.225)
sementeAleatoria = 42
amostraPorClasse = int(os.getenv("SAMPLEPERCLASS") or 3000)
proporcaoTeste = 0.15
proporcaoValidacao = 0.15
nomesClasses = ("noCrack", "crack")
epocasCabeca = int(os.getenv("HEADEPOCHS") or 2)
epocasAjuste = int(os.getenv("TUNEEPOCHS") or 6)
tamanhoLote = 64
usarPesosPreTreinados = (os.getenv("PRETRAINED") or "true").lower() == "true"
quantidadeGrupos = int(os.getenv("CLUSTERS") or 150)
