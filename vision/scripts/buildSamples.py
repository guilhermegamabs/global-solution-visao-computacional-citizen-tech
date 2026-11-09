import cv2
import numpy as np
import pandas as pd
from PIL import Image

from vision import settings

lado = 227
pastaImagens = settings.raizProjeto / "samples" / "imagens"
pastaVideos = settings.raizProjeto / "samples" / "videos"


def sortearRecortes(tabela, rotulo, quantidade, gerador):
    caminhos = tabela[tabela["label"] == rotulo]["path"].tolist()
    escolhidos = gerador.choice(len(caminhos), size=quantidade, replace=False)
    return [Image.open(caminhos[indice]).convert("RGB") for indice in escolhidos]


def montarMosaico(colunas, linhas, fissuras, tabela, gerador):
    total = colunas * linhas
    limpos = sortearRecortes(tabela, 0, total, gerador)
    rachados = sortearRecortes(tabela, 1, max(len(fissuras), 1), gerador)
    tela = Image.new("RGB", (colunas * lado, linhas * lado))
    for posicao in range(total):
        linha, coluna = divmod(posicao, colunas)
        tela.paste(limpos[posicao], (coluna * lado, linha * lado))
    for recorte, (linha, coluna) in zip(rachados, fissuras):
        tela.paste(recorte, (coluna * lado, linha * lado))
    return tela


def gravarVideo(mosaico, caminho, largura, altura, quadrosPorSegundo, quadros):
    matriz = cv2.cvtColor(np.asarray(mosaico), cv2.COLOR_RGB2BGR)
    escritor = cv2.VideoWriter(str(caminho), cv2.VideoWriter_fourcc(*"mp4v"), quadrosPorSegundo, (largura, altura))
    deslocamentoMaximo = matriz.shape[1] - largura
    for quadro in range(quadros):
        inicio = int(deslocamentoMaximo * quadro / (quadros - 1))
        escritor.write(matriz[:altura, inicio:inicio + largura])
    escritor.release()


def executar():
    pastaImagens.mkdir(parents=True, exist_ok=True)
    pastaVideos.mkdir(parents=True, exist_ok=True)
    tabela = pd.read_csv(settings.arquivoDivisao)
    teste = tabela[tabela["split"] == "test"]
    gerador = np.random.default_rng(settings.sementeAleatoria)

    montarMosaico(6, 4, [(1, 2), (1, 3), (2, 3), (2, 4)], teste, gerador).save(
        pastaImagens / "paredeComFissura.jpg", quality=92
    )
    montarMosaico(6, 4, [], teste, gerador).save(pastaImagens / "paredeSemFissura.jpg", quality=92)

    faixa = montarMosaico(10, 3, [(1, 4), (1, 5), (2, 5), (1, 6)], teste, gerador)
    faixa.save(pastaImagens / "faixaDeParede.jpg", quality=92)
    gravarVideo(faixa, pastaVideos / "varreduraDaFachada.mp4", 640, 480, 15.0, 150)
    print("Exemplos gerados em samples/")


executar()
