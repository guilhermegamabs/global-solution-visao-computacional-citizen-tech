import random
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from vision import settings

extensoesValidas = {".jpg", ".jpeg", ".png"}
ladoMiniatura = 16
componentesPca = 40


def localizarPastasClasses():
    pastas = {}
    for caminho in settings.pastaBruta.rglob("*"):
        if caminho.is_dir() and caminho.name.lower() in ("positive", "negative"):
            pastas[caminho.name.lower()] = caminho
    if len(pastas) != 2:
        raise SystemExit(f"Pastas Positive e Negative não encontradas em {settings.pastaBruta}")
    return pastas


def listarImagens(pasta):
    return sorted(caminho for caminho in pasta.iterdir() if caminho.suffix.lower() in extensoesValidas)


def amostrarClasse(imagens, quantidade, sorteio):
    if len(imagens) <= quantidade:
        return list(imagens)
    return sorteio.sample(imagens, quantidade)


def montarTabela(pastas):
    sorteio = random.Random(settings.sementeAleatoria)
    linhas = []
    for nomePasta, rotulo in (("negative", 0), ("positive", 1)):
        imagens = amostrarClasse(listarImagens(pastas[nomePasta]), settings.amostraPorClasse, sorteio)
        for caminho in imagens:
            linhas.append({"path": caminho.resolve().as_posix(), "label": rotulo})
    return pd.DataFrame(linhas)


def extrairCaracteristicas(caminhos):
    vetores = []
    for caminho in caminhos:
        miniatura = Image.open(caminho).convert("RGB").resize((ladoMiniatura, ladoMiniatura), Image.Resampling.BILINEAR)
        vetores.append(np.asarray(miniatura, dtype=np.float32).ravel() / 255.0)
    return np.stack(vetores)


def agruparPorAparencia(tabela):
    caracteristicas = StandardScaler().fit_transform(extrairCaracteristicas(tabela["path"]))
    reduzidas = PCA(n_components=componentesPca, random_state=settings.sementeAleatoria).fit_transform(caracteristicas)
    modelo = KMeans(n_clusters=settings.quantidadeGrupos, n_init=3, random_state=settings.sementeAleatoria)
    tabela = tabela.copy()
    tabela["group"] = [f"g{numero:03d}" for numero in modelo.fit_predict(reduzidas)]
    return tabela


def dividirPorGrupo(tabela):
    divisoesTeste = round(1 / settings.proporcaoTeste)
    primeira = StratifiedGroupKFold(n_splits=divisoesTeste, shuffle=True, random_state=settings.sementeAleatoria)
    indicesRestante, indicesTeste = next(primeira.split(tabela, tabela["label"], tabela["group"]))
    restante = tabela.iloc[indicesRestante]

    divisoesValidacao = round((1 - settings.proporcaoTeste) / settings.proporcaoValidacao)
    segunda = StratifiedGroupKFold(n_splits=divisoesValidacao, shuffle=True, random_state=settings.sementeAleatoria)
    indicesTreino, indicesValidacao = next(segunda.split(restante, restante["label"], restante["group"]))

    tabela = tabela.copy()
    coluna = tabela.columns.get_loc("split")
    tabela.iloc[indicesRestante[indicesTreino], coluna] = "train"
    tabela.iloc[indicesRestante[indicesValidacao], coluna] = "validation"
    tabela.iloc[indicesTeste, coluna] = "test"
    return tabela


def contarGruposCompartilhados(tabela):
    gruposPorDivisao = tabela.groupby("split")["group"].agg(set)
    nomes = list(gruposPorDivisao.index)
    compartilhados = 0
    for posicao, primeiro in enumerate(nomes):
        for segundo in nomes[posicao + 1:]:
            compartilhados += len(gruposPorDivisao[primeiro] & gruposPorDivisao[segundo])
    return compartilhados


def executar():
    pastas = localizarPastasClasses()
    tabela = agruparPorAparencia(montarTabela(pastas))
    tabela["split"] = ""
    tabela = dividirPorGrupo(tabela)
    settings.arquivoDivisao.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(settings.arquivoDivisao, index=False)

    print(f"Imagens na amostra: {len(tabela)}")
    print(f"Grupos visuais: {tabela['group'].nunique()}")
    print(f"Grupos compartilhados entre divisões: {contarGruposCompartilhados(tabela)}")
    for divisao in ("train", "validation", "test"):
        recorte = tabela[tabela["split"] == divisao]
        contagem = Counter(recorte["label"])
        print(f"{divisao}: {len(recorte)} imagens, {recorte['group'].nunique()} grupos, sem fissura {contagem[0]}, com fissura {contagem[1]}")
    print(f"Tabela salva em {settings.arquivoDivisao}")


executar()
