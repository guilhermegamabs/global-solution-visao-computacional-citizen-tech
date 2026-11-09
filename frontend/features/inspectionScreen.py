import base64

import plotly.graph_objects as go
import streamlit as st

from features.apiErrors import mostrarErroApi
from providers.apiClient import ApiError, buscarModeloInspecao, inspecionarImagem, inspecionarVideo
from state.sessionState import definirInspecao, obterInspecao, obterToken
from ui.components import (
    renderizarAviso,
    renderizarCartaoMetrica,
    renderizarEstadoVazio,
    renderizarFaixaErro,
    renderizarSeloInspecao,
)
from ui.theme import azulPrimario, azulProfundo

tiposImagem = ["jpg", "jpeg", "png"]
tiposVideo = ["mp4", "mov", "avi"]


def decodificarImagem(texto):
    return base64.b64decode(texto)


def formatarPorcentagem(valor):
    return f"{valor * 100:.1f}%".replace(".", ",")


def formatarNumero(valor, casas=1):
    return f"{valor:.{casas}f}".replace(".", ",")


def renderizarLimiar(modelo):
    padrao = float(modelo.get("defaultThreshold", 0.5))
    return st.slider(
        "Limiar de confiança",
        min_value=0.10,
        max_value=0.95,
        value=padrao,
        step=0.05,
        help="Valores baixos sinalizam mais locais, mas aumentam alarmes falsos. Valores altos fazem o contrário.",
    )


def renderizarMetricasModelo(modelo):
    metricas = modelo.get("metrics")
    if not metricas:
        return
    teste = metricas.get("test", {})
    with st.expander("Qualidade do modelo no conjunto de teste"):
        colunas = st.columns(4)
        rotulos = (("Precisão", "precision"), ("Recall", "recall"), ("F1", "f1"), ("Acurácia", "accuracy"))
        for coluna, (rotulo, chave) in zip(colunas, rotulos):
            with coluna:
                renderizarCartaoMetrica(rotulo, formatarPorcentagem(teste.get(chave, 0)))
        st.caption(
            "Medido em recortes de concreto de laboratório. Em fotos reais de fachadas e calçadas o resultado pode ser pior."
        )


def processarImagem(arquivo, limiar):
    try:
        with st.spinner("Analisando a imagem..."):
            resultado = inspecionarImagem(arquivo.name, arquivo.getvalue(), arquivo.type or "image/jpeg", limiar, obterToken())
    except ApiError as erro:
        mostrarErroApi(erro)
        return
    definirInspecao("imagem", {"nomeArquivo": arquivo.name, "original": arquivo.getvalue(), "resultado": resultado})


def renderizarResultadoImagem(dados):
    resultado = dados["resultado"]
    renderizarSeloInspecao(resultado["needsInspection"], grande=True)
    st.write("")

    colunas = st.columns(4)
    with colunas[0]:
        renderizarCartaoMetrica("Confiança", formatarPorcentagem(resultado["confidence"]), "do resultado mostrado")
    with colunas[1]:
        renderizarCartaoMetrica("Prob. de fissura", formatarPorcentagem(resultado["crackProbability"]), "maior valor da imagem")
    with colunas[2]:
        renderizarCartaoMetrica("Blocos sinalizados", f'{resultado["flaggedTiles"]} de {resultado["totalTiles"]}', "regiões da foto")
    with colunas[3]:
        renderizarCartaoMetrica("Tempo de análise", f'{formatarNumero(resultado["latencyMs"])} ms', f'{resultado["imageWidth"]}×{resultado["imageHeight"]} px')

    colunaOriginal, colunaAnalise = st.columns(2, gap="large")
    with colunaOriginal:
        st.markdown("**Foto enviada**")
        st.image(dados["original"], width="stretch")
    with colunaAnalise:
        st.markdown("**Regiões que merecem inspeção**")
        st.image(decodificarImagem(resultado["annotatedImage"]), width="stretch")
        st.caption("Blocos em vermelho passaram do limiar de confiança escolhido.")

    renderizarAviso(resultado["disclaimer"])


def renderizarAbaImagem(limiar):
    arquivo = st.file_uploader("Foto da fachada, parede ou calçada", type=tiposImagem, key="inspecaoArquivoImagem")
    if st.button("Analisar foto", type="primary", disabled=arquivo is None):
        processarImagem(arquivo, limiar)

    dados = obterInspecao("imagem")
    if dados:
        renderizarResultadoImagem(dados)
    else:
        renderizarEstadoVazio("📷", "Nenhuma foto analisada", "Envie uma foto e clique em Analisar foto.")


def processarVideo(arquivo, limiar):
    try:
        with st.spinner("Processando o vídeo quadro a quadro. Isso pode levar alguns segundos..."):
            resultado = inspecionarVideo(arquivo.name, arquivo.getvalue(), arquivo.type or "video/mp4", limiar, obterToken())
    except ApiError as erro:
        mostrarErroApi(erro)
        return
    definirInspecao("video", {"nomeArquivo": arquivo.name, "resultado": resultado})


def montarGraficoLinhaDoTempo(resultado):
    pontos = resultado["timeline"]
    tempos = [ponto["timeSeconds"] for ponto in pontos]
    probabilidades = [ponto["crackProbability"] for ponto in pontos]
    cores = ["#D32F2F" if ponto["flagged"] else azulPrimario for ponto in pontos]

    figura = go.Figure()
    figura.add_trace(
        go.Scatter(
            x=tempos,
            y=probabilidades,
            mode="lines+markers",
            line={"color": azulProfundo, "width": 2},
            marker={"color": cores, "size": 9},
            name="Probabilidade de fissura",
            hovertemplate="%{x:.1f} s: %{y:.0%}<extra></extra>",
        )
    )
    figura.add_hline(
        y=resultado["threshold"],
        line_dash="dash",
        line_color="#D32F2F",
        annotation_text="Limiar",
        annotation_position="top left",
    )
    figura.update_layout(
        height=300,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        xaxis_title="Tempo do vídeo (s)",
        yaxis_title="Probabilidade de fissura",
        yaxis={"range": [0, 1], "tickformat": ".0%"},
        showlegend=False,
    )
    return figura


def renderizarResultadoVideo(dados):
    resultado = dados["resultado"]
    renderizarSeloInspecao(resultado["needsInspection"], grande=True)
    st.write("")

    colunas = st.columns(4)
    with colunas[0]:
        renderizarCartaoMetrica("Quadros analisados", resultado["framesAnalyzed"], f'vídeo de {formatarNumero(resultado["durationSeconds"])} s')
    with colunas[1]:
        renderizarCartaoMetrica("Quadros sinalizados", f'{resultado["framesFlagged"]} ({formatarPorcentagem(resultado["flaggedRatio"])})')
    with colunas[2]:
        renderizarCartaoMetrica("Latência média", f'{formatarNumero(resultado["averageLatencyMs"])} ms', "por quadro")
    with colunas[3]:
        renderizarCartaoMetrica("Taxa de processamento", f'{formatarNumero(resultado["processingFps"])} FPS', "quadros analisados por segundo")

    st.markdown("**Linha do tempo do monitoramento**")
    st.plotly_chart(montarGraficoLinhaDoTempo(resultado), width="stretch")

    previas = resultado["previews"]
    if previas:
        st.markdown("**Quadros com maior chance de fissura**")
        colunasPrevias = st.columns(len(previas))
        for coluna, previa in zip(colunasPrevias, previas):
            with coluna:
                st.image(decodificarImagem(previa["image"]), width="stretch")
                st.caption(f'{formatarNumero(previa["timeSeconds"])} s · {formatarPorcentagem(previa["crackProbability"])}')
    else:
        st.info("Nenhum quadro passou do limiar escolhido.", icon="✅")

    renderizarAviso(resultado["disclaimer"])


def renderizarAbaVideo(limiar):
    arquivo = st.file_uploader("Vídeo curto da fachada ou do trajeto", type=tiposVideo, key="inspecaoArquivoVideo")
    st.caption("Analisamos 1 quadro a cada 5, em até 90 quadros. Use vídeos de até 60 MB.")
    if st.button("Analisar vídeo", type="primary", disabled=arquivo is None):
        processarVideo(arquivo, limiar)

    dados = obterInspecao("video")
    if dados:
        renderizarResultadoVideo(dados)
    else:
        renderizarEstadoVazio("🎞️", "Nenhum vídeo analisado", "Envie um vídeo curto para simular o monitoramento contínuo.")


def renderizarInspecao():
    st.subheader("Inspeção predial de unidades de saúde")
    st.caption(
        "Envie uma foto ou um vídeo curto de uma UBS ou UPA. O app sinaliza possíveis fissuras no concreto "
        "para ajudar a priorizar a visita de um profissional."
    )

    try:
        modelo = buscarModeloInspecao(obterToken())
    except ApiError as erro:
        mostrarErroApi(erro)
        return

    if not modelo.get("available"):
        renderizarFaixaErro("O modelo de inspeção ainda não foi instalado neste ambiente.")
        return

    limiar = renderizarLimiar(modelo)
    renderizarMetricasModelo(modelo)

    abaFoto, abaVideo = st.tabs(["📷 Foto", "🎞️ Vídeo curto"])
    with abaFoto:
        renderizarAbaImagem(limiar)
    with abaVideo:
        renderizarAbaVideo(limiar)
