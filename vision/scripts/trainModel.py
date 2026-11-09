import copy
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from vision import settings

dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

normalizacao = transforms.Normalize(settings.mediaImagenet, settings.desvioImagenet)

transformacaoTreino = transforms.Compose(
    [
        transforms.Resize((settings.tamanhoEntrada, settings.tamanhoEntrada)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
        transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 1.5))], p=0.2),
        transforms.ToTensor(),
        normalizacao,
    ]
)

transformacaoAvaliacao = transforms.Compose(
    [
        transforms.Resize((settings.tamanhoEntrada, settings.tamanhoEntrada)),
        transforms.ToTensor(),
        normalizacao,
    ]
)


class ConjuntoImagens(Dataset):
    def __init__(self, tabela, transformacao):
        self.caminhos = tabela["path"].tolist()
        self.rotulos = tabela["label"].tolist()
        self.transformacao = transformacao
        self.imagens = [Image.open(caminho).convert("RGB") for caminho in self.caminhos]

    def __len__(self):
        return len(self.caminhos)

    def __getitem__(self, posicao):
        return self.transformacao(self.imagens[posicao]), self.rotulos[posicao]


def criarModelo():
    pesos = models.MobileNet_V3_Small_Weights.DEFAULT if settings.usarPesosPreTreinados else None
    modelo = models.mobilenet_v3_small(weights=pesos)
    entradas = modelo.classifier[3].in_features
    modelo.classifier[3] = nn.Linear(entradas, 2)
    return modelo.to(dispositivo)


def congelarBase(modelo, congelar):
    for parametro in modelo.features.parameters():
        parametro.requires_grad = not congelar


def treinarEpoca(modelo, carregador, otimizador, funcaoPerda):
    modelo.train()
    perdaTotal = 0.0
    for lote, rotulos in carregador:
        lote = lote.to(dispositivo)
        rotulos = rotulos.to(dispositivo)
        otimizador.zero_grad()
        perda = funcaoPerda(modelo(lote), rotulos)
        perda.backward()
        otimizador.step()
        perdaTotal += perda.item() * len(rotulos)
    return perdaTotal / len(carregador.dataset)


def avaliar(modelo, carregador, funcaoPerda):
    modelo.eval()
    perdaTotal = 0.0
    previsoes = []
    reais = []
    with torch.no_grad():
        for lote, rotulos in carregador:
            lote = lote.to(dispositivo)
            rotulos = rotulos.to(dispositivo)
            saidas = modelo(lote)
            perdaTotal += funcaoPerda(saidas, rotulos).item() * len(rotulos)
            previsoes.extend(saidas.argmax(dim=1).cpu().tolist())
            reais.extend(rotulos.cpu().tolist())
    acuracia = float(np.mean(np.array(previsoes) == np.array(reais)))
    return perdaTotal / len(carregador.dataset), acuracia, f1_score(reais, previsoes)


def executarFase(nome, modelo, carregadores, taxa, epocas, estado):
    funcaoPerda = nn.CrossEntropyLoss()
    parametros = [parametro for parametro in modelo.parameters() if parametro.requires_grad]
    otimizador = torch.optim.AdamW(parametros, lr=taxa, weight_decay=1e-4)
    for epoca in range(1, epocas + 1):
        inicio = time.perf_counter()
        perdaTreino = treinarEpoca(modelo, carregadores["train"], otimizador, funcaoPerda)
        perdaValidacao, acuracia, f1 = avaliar(modelo, carregadores["validation"], funcaoPerda)
        duracao = time.perf_counter() - inicio
        estado["historico"].append(
            {"phase": nome, "epoch": epoca, "trainLoss": perdaTreino, "valLoss": perdaValidacao, "valAccuracy": acuracia, "valF1": f1}
        )
        marcador = ""
        if f1 >= estado["melhorF1"]:
            estado["melhorF1"] = f1
            estado["melhorEstado"] = copy.deepcopy(modelo.state_dict())
            marcador = " *"
        print(
            f"{nome} {epoca}/{epocas} perda treino {perdaTreino:.4f} perda val {perdaValidacao:.4f} "
            f"acurácia val {acuracia:.4f} F1 val {f1:.4f} ({duracao:.0f}s){marcador}"
        )


def exportarOnnx(modelo):
    settings.pastaModelos.mkdir(parents=True, exist_ok=True)
    modelo = modelo.to("cpu").eval()
    exemplo = torch.zeros(1, 3, settings.tamanhoEntrada, settings.tamanhoEntrada)
    torch.onnx.export(
        modelo,
        exemplo,
        str(settings.caminhoOnnx),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"Modelo exportado para {settings.caminhoOnnx}")


def executar():
    torch.manual_seed(settings.sementeAleatoria)
    tabela = pd.read_csv(settings.arquivoDivisao)
    print(f"Dispositivo: {dispositivo}")

    carregadores = {}
    for divisao, transformacao in (("train", transformacaoTreino), ("validation", transformacaoAvaliacao)):
        conjunto = ConjuntoImagens(tabela[tabela["split"] == divisao], transformacao)
        carregadores[divisao] = DataLoader(conjunto, batch_size=settings.tamanhoLote, shuffle=divisao == "train", num_workers=0)

    modelo = criarModelo()
    estado = {"melhorF1": -1.0, "melhorEstado": None, "historico": []}

    congelarBase(modelo, True)
    executarFase("cabeça", modelo, carregadores, 1e-3, settings.epocasCabeca, estado)
    congelarBase(modelo, False)
    executarFase("ajuste", modelo, carregadores, 1e-4, settings.epocasAjuste, estado)

    modelo.load_state_dict(estado["melhorEstado"])
    settings.pastaExecucoes.mkdir(parents=True, exist_ok=True)
    torch.save(modelo.state_dict(), settings.caminhoPesos)
    pd.DataFrame(estado["historico"]).to_csv(settings.pastaExecucoes / "history.csv", index=False)
    exportarOnnx(modelo)
    print(f"Melhor F1 de validação: {estado['melhorF1']:.4f}")


executar()
