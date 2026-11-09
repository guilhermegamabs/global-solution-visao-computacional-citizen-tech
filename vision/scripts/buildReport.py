import base64
import io
import json
import sys
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as ImagemPil
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from vision import settings

sys.path.insert(0, str(settings.raizProjeto / "backend"))

from app.pipelines.inspectionPipeline import analisarImagem, analisarVideo
from app.providers.classifierProvider import classificarLote

azulProfundo = colors.HexColor("#073B6B")
azulPrimario = colors.HexColor("#0B5FA5")
azulSuave = colors.HexColor("#EAF3FC")
caminhoRelatorio = settings.pastaRelatorio / "relatorioInspecaoPredial.pdf"
larguraUtil = A4[0] - 4 * cm

estilos = getSampleStyleSheet()
estiloTitulo = ParagraphStyle("titulo", parent=estilos["Title"], textColor=azulProfundo, fontSize=20, leading=24, spaceAfter=4)
estiloSubtitulo = ParagraphStyle("subtitulo", parent=estilos["Normal"], textColor=azulPrimario, fontSize=11, alignment=1, spaceAfter=10)
estiloSecao = ParagraphStyle("secao", parent=estilos["Heading2"], textColor=azulProfundo, fontSize=13, spaceBefore=12, spaceAfter=5)
estiloCorpo = ParagraphStyle("corpo", parent=estilos["Normal"], fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=5)
estiloLista = ParagraphStyle("lista", parent=estiloCorpo, leftIndent=14, bulletIndent=2, spaceAfter=2)
estiloLegenda = ParagraphStyle("legenda", parent=estilos["Normal"], fontSize=8.5, leading=11, textColor=colors.HexColor("#3E5B78"), alignment=1, spaceAfter=8)
estiloCelula = ParagraphStyle("celula", parent=estilos["Normal"], fontSize=9, leading=11)
estiloCabecalho = ParagraphStyle("cabecalho", parent=estiloCelula, textColor=colors.white, fontName="Helvetica-Bold")


def paragrafo(texto):
    return Paragraph(texto, estiloCorpo)


def marcador(texto):
    return Paragraph(texto, estiloLista, bulletText="•")


def porcentagem(valor):
    return f"{valor * 100:.2f}".replace(".", ",") + "%"


def numero(valor, casas=2):
    return f"{valor:.{casas}f}".replace(".", ",")


def montarTabela(cabecalho, linhas, larguras):
    dados = [[Paragraph(texto, estiloCabecalho) for texto in cabecalho]]
    dados += [[Paragraph(str(celula), estiloCelula) for celula in linha] for linha in linhas]
    tabela = Table(dados, colWidths=larguras, repeatRows=1)
    tabela.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), azulProfundo),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, azulSuave]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C9DDF0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return tabela


def figura(origem, largura, legenda):
    if isinstance(origem, (bytes, bytearray)):
        origem = io.BytesIO(origem)
    imagem = ImagemPil.open(origem)
    proporcao = imagem.height / imagem.width
    origem.seek(0) if hasattr(origem, "seek") else None
    return KeepTogether([Image(origem, width=largura, height=largura * proporcao), Paragraph(legenda, estiloLegenda)])


def recortarBorda(caminho):
    imagem = ImagemPil.open(caminho).convert("RGB")
    fundo = ImagemPil.new("RGB", imagem.size, (255, 255, 255))
    from PIL import ImageChops

    caixa = ImageChops.difference(imagem, fundo).getbbox()
    saida = io.BytesIO()
    (imagem.crop(caixa) if caixa else imagem).save(saida, format="PNG")
    return saida.getvalue()


def graficoLinhaDoTempo(resultado):
    figuraBase, eixo = plt.subplots(figsize=(7, 2.8))
    tempos = [ponto.timeSeconds for ponto in resultado.timeline]
    probabilidades = [ponto.crackProbability for ponto in resultado.timeline]
    eixo.plot(tempos, probabilidades, color="#073B6B", marker="o", markersize=4)
    eixo.axhline(resultado.threshold, color="#D32F2F", linestyle="--", label=f"Limiar {resultado.threshold:.2f}")
    eixo.set_ylim(0, 1.05)
    eixo.set_xlabel("Tempo do vídeo (s)")
    eixo.set_ylabel("Probabilidade de fissura")
    eixo.legend(loc="center left")
    eixo.grid(alpha=0.3)
    figuraBase.tight_layout()
    saida = io.BytesIO()
    figuraBase.savefig(saida, format="png", dpi=170)
    plt.close(figuraBase)
    return saida.getvalue()


def rodape(tela, documento):
    tela.saveState()
    tela.setFont("Helvetica", 8)
    tela.setFillColor(colors.HexColor("#3E5B78"))
    tela.drawString(2 * cm, 1.2 * cm, "CitizenTech · Orienta Saúde · Inspeção predial com visão computacional")
    tela.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Página {documento.page}")
    tela.restoreState()


def montarHistorias(metricas, benchmark, exemploImagem, exemploVideo):
    teste = metricas["test"]
    padrao = metricas["testAtDefaultThreshold"]
    amostra = metricas["sample"]
    limiar = metricas["recommendedThreshold"]
    matriz = teste["confusionMatrix"]
    unitario = benchmark["singleImage224"]
    lote = benchmark["tileBatch"]
    foto = benchmark["fullPhotoPipeline"]
    equipamento = benchmark["equipment"]
    historias = []

    historias.append(Paragraph("Inspeção predial com visão computacional", estiloTitulo))
    historias.append(Paragraph(f"Relatório técnico · Global Solution 2026/2 · CitizenTech · {date.today().strftime('%d/%m/%Y')}", estiloSubtitulo))
    historias.append(paragrafo("<b>Integrantes:</b> [preencher com os nomes completos]"))

    historias.append(Paragraph("1. Objetivo e limite de uso", estiloSecao))
    historias.append(
        paragrafo(
            "O módulo estende o protótipo Orienta Saúde com uma aba de inspeção predial. O usuário envia uma foto ou um vídeo curto "
            "de uma UBS ou UPA, e o sistema indica se há indício de fissura no concreto, mostrando as regiões suspeitas e a "
            "confiança do modelo. A finalidade é <b>apoiar a triagem de locais para inspeção</b>."
        )
    )
    historias.append(
        paragrafo(
            "<b>Limite:</b> a presença de fissura na imagem, isoladamente, não permite determinar a gravidade do dano nem a "
            "segurança estrutural da edificação. Esse aviso aparece em todo resultado, e a avaliação final é de um profissional habilitado."
        )
    )

    historias.append(Paragraph("2. Dados e preparo", estiloSecao))
    historias.append(
        paragrafo(
            "Base escolhida: <i>Concrete Crack Images for Classification</i> (Mendeley Data), com 20 mil imagens de concreto com "
            "fissura e 20 mil sem fissura, de 227 x 227 pixels. Para caber no prazo de quatro horas, usamos uma amostra balanceada de "
            f"{settings.amostraPorClasse} imagens por classe ({amostra['train'] + amostra['validation'] + amostra['test']} no total)."
        )
    )
    historias.append(marcador("<b>Redimensionamento e normalização:</b> 224 x 224 pixels, com média e desvio-padrão do ImageNet, exigidos pelos pesos pré-treinados."))
    historias.append(marcador("<b>Aumento de dados (só no treino):</b> espelhamento, rotação de até 20 graus, variação de brilho, contraste e saturação, e desfoque leve, para tolerar sombra e iluminação diferentes."))
    historias.append(
        marcador(
            f"<b>Divisão:</b> treino {amostra['train']}, validação {amostra['validation']} e teste {amostra['test']} imagens, "
            "estratificada por classe e sem recortes do mesmo grupo em conjuntos diferentes."
        )
    )
    historias.append(
        paragrafo(
            "<b>Como evitamos recortes da mesma origem em conjuntos diferentes.</b> Os arquivos da base têm nomes como 00001.jpg e "
            "00001_1.jpg, com um número único por arquivo, então o nome não informa de qual foto grande o recorte saiu. "
            "Conferimos também que a numeração sequencial não ajuda: nas imagens com fissura, vizinhas na numeração são mais "
            "parecidas entre si do que pares aleatórios, mas nas imagens sem fissura não há relação nenhuma. Por isso agrupamos as imagens por "
            f"aparência (miniatura 16 x 16 em cor, PCA e KMeans com {settings.quantidadeGrupos} grupos), aproximando a superfície e a "
            "iluminação de origem, e dividimos por grupo com <i>StratifiedGroupKFold</i>. O <i>prepareData</i> confirma que "
            "nenhum grupo aparece em mais de um conjunto. O teste mede, portanto, o desempenho em superfícies que o modelo não viu."
        )
    )

    historias.append(Paragraph("3. Modelo e justificativa da arquitetura", estiloSecao))
    historias.append(
        paragrafo(
            "Usamos <b>MobileNetV3-Small pré-treinada no ImageNet</b>, com a última camada trocada por duas classes (sem fissura e com "
            "fissura). O treino tem duas fases: primeiro só a cabeça de classificação (2 épocas, taxa 1e-3), depois o ajuste fino de toda a "
            "rede (6 épocas, taxa 1e-4), com AdamW. Guardamos a época de maior F1 na validação e exportamos para ONNX."
        )
    )
    historias.append(marcador("<b>Leve:</b> cerca de 2,5 milhões de parâmetros. Roda em tempo real na CPU, que é o que um servidor gratuito oferece, e dispensa GPU no deploy."))
    historias.append(marcador("<b>Transferência de aprendizado:</b> com poucos milhares de imagens, reaproveitar bordas e texturas do ImageNet converge em poucas épocas e evita sobreajuste."))
    historias.append(marcador("<b>ONNX Runtime:</b> o servidor não precisa do PyTorch, o que tende a reduzir o tamanho da imagem Docker e o tempo de inicialização."))
    historias.append(marcador("<b>Grade 3 x 3:</b> a base é de recortes pequenos de superfície, mas fotos reais são maiores. Classificar a foto inteira e nove blocos usa o modelo na escala em que foi treinado e mostra onde está a suspeita. É localização aproximada, não detecção com caixas."))

    historias.append(Paragraph("4. Resultados", estiloSecao))
    historias.append(
        paragrafo(
            f"O limiar de confiança foi escolhido <b>na validação</b>, pelo maior F2, que dá mais peso ao recall do que à precisão. Na validação, o F2 ficou empatado no máximo entre os limiares 0,40 e 0,85, e adotamos o menor deles: {numero(limiar)}. "
            "O conjunto de teste só mediu o resultado final. A tabela mostra o teste com esse limiar e com o limiar neutro de 0,50."
        )
    )
    historias.append(
        montarTabela(
            ["Limiar", "Precisão", "Recall", "F1", "Acurácia", "Matriz (VN / FP / FN / VP)"],
            [
                [numero(teste["threshold"]) + " (escolhido)", porcentagem(teste["precision"]), porcentagem(teste["recall"]), porcentagem(teste["f1"]), porcentagem(teste["accuracy"]), f"{matriz[0][0]} / {matriz[0][1]} / {matriz[1][0]} / {matriz[1][1]}"],
                [numero(padrao["threshold"]), porcentagem(padrao["precision"]), porcentagem(padrao["recall"]), porcentagem(padrao["f1"]), porcentagem(padrao["accuracy"]), f"{padrao['confusionMatrix'][0][0]} / {padrao['confusionMatrix'][0][1]} / {padrao['confusionMatrix'][1][0]} / {padrao['confusionMatrix'][1][1]}"],
            ],
            [3.0 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 5.2 * cm],
        )
    )
    historias.append(Spacer(1, 6))
    historias.append(figura(settings.pastaPrints / "confusionMatrix.png", 7 * cm, f"Figura 1. Matriz de confusão no conjunto de teste ({amostra['test']} imagens), limiar {numero(limiar)}."))
    historias.append(figura(settings.pastaPrints / "trainingCurves.png", 13 * cm, "Figura 2. Perda e F1 de validação por época (eixos ampliados). As duas primeiras épocas são da fase só da cabeça."))
    historias.append(figura(settings.pastaPrints / "thresholdSweep.png", 11 * cm, "Figura 3. Precisão, recall e F1 na validação para cada limiar. O eixo vertical está ampliado: as diferenças são de décimos de ponto percentual."))
    historias.append(
        paragrafo(
            "<b>Leitura honesta dos números.</b> O F1 de validação chegou a 1,0 e o de teste ficou em torno de 99,7%. Essa base é conhecida por ser "
            "fácil: as fissuras são nítidas e o fundo é concreto de laboratório. Os valores <b>não representam</b> o desempenho em fachadas reais de UBS e UPA, "
            "com pintura, azulejo, manchas, vegetação e fissuras finas. Não medimos esse cenário por falta de fotos reais anotadas, e ele é o principal passo seguinte."
        )
    )

    historias.append(Paragraph("5. Análise de acertos e erros", estiloSecao))
    historias.append(
        paragrafo(
            f"No teste com limiar {numero(limiar)} houve <b>{matriz[0][1]} falsos positivos e {matriz[1][0]} falsos negativos</b>. Nenhuma fissura do teste passou despercebida, "
            "o que atende ao critério de triagem, em que é pior deixar passar uma fissura do que mandar verificar um local sem dano."
        )
    )
    historias.append(figura(recortarBorda(settings.pastaPrints / "falsePositives.png"), 13 * cm, "Figura 4. Os falsos positivos do teste, com a probabilidade de fissura prevista."))
    historias.append(
        paragrafo(
            "Olhando as três imagens, duas têm marcas que parecem fissuras: uma linha escura nítida cruzando a superfície (previsão 1,00) e uma fissura fraca e fina "
            "(previsão 0,42). A terceira tem manchas e variação de claridade (previsão 0,58). Ou seja, parte desses erros provavelmente é <b>ruído de rótulo da própria base</b>, "
            "e não falha do modelo. Essa é uma leitura visual nossa, sem confirmação por especialista."
        )
    )
    historias.append(paragrafo("Para medir a sensibilidade a sombra, iluminação e textura, aplicamos perturbações sintéticas às imagens de teste e repetimos a avaliação, sempre com o mesmo limiar."))
    linhasRobustez = [
        [item["name"], porcentagem(item["precision"]), porcentagem(item["recall"]), porcentagem(item["f1"])]
        for item in metricas["robustness"]
    ]
    historias.append(montarTabela(["Perturbação", "Precisão", "Recall", "F1"], linhasRobustez, [5 * cm, 3 * cm, 3 * cm, 3 * cm]))
    historias.append(Spacer(1, 6))
    piorRecall = min(metricas["robustness"], key=lambda item: item["recall"])
    piorPrecisao = min(metricas["robustness"], key=lambda item: item["precision"])
    historias.append(
        paragrafo(
            f"O modelo se manteve estável diante de escurecimento, sombra lateral e baixo contraste, com treino que já incluía variação de brilho e contraste. "
            f"A maior perda de recall veio de <b>{piorRecall['name'].lower()}</b> ({porcentagem(piorRecall['recall'])}) e a maior perda de precisão veio de <b>{piorPrecisao['name'].lower()}</b> ({porcentagem(piorPrecisao['precision'])}). "
            "Ruído e imagem clara reduziram o recall, e o desfoque reduziu a precisão. Não investigamos a causa. "
            "São perturbações sintéticas aplicadas a imagens de laboratório, e não substituem um teste com fotos de campo."
        )
    )

    historias.append(Paragraph("6. Integração ao CitizenTech", estiloSecao))
    historias.append(
        paragrafo(
            "A aba <b>Inspeção predial</b> do Orienta Saúde recebe uma foto ou um vídeo curto, exige login, e chama a API FastAPI (rotas <i>/inspection/image</i> e "
            "<i>/inspection/video</i>). O limiar de confiança pode ser ajustado na tela. Fotos e vídeos são processados em memória e não são gravados."
        )
    )
    historias.append(figura(base64.b64decode(exemploImagem.annotatedImage), 11 * cm, f"Figura 5. Exemplo de foto: {exemploImagem.flaggedTiles} de {exemploImagem.totalTiles} blocos sinalizados, probabilidade de fissura {porcentagem(exemploImagem.crackProbability)}. O exemplo é um mosaico de recortes do conjunto de teste."))
    historias.append(
        paragrafo(
            f"Para simular o monitoramento urbano, o vídeo é lido quadro a quadro (1 a cada 5, até 90 quadros). No vídeo de demonstração, de {numero(exemploVideo.durationSeconds, 1)} s, "
            f"foram analisados {exemploVideo.framesAnalyzed} quadros e {exemploVideo.framesFlagged} foram sinalizados. A linha do tempo mostra o trecho em que a fissura entra e sai do quadro. "
            f"O vídeo é considerado suspeito quando pelo menos dois quadros passam do limiar, para reduzir alarmes isolados."
        )
    )
    historias.append(figura(graficoLinhaDoTempo(exemploVideo), 12 * cm, "Figura 6. Probabilidade de fissura por quadro no vídeo de demonstração."))

    historias.append(Paragraph("7. Desempenho", estiloSecao))
    historias.append(
        paragrafo(
            f"<b>Equipamento:</b> {equipamento['processor']} ({equipamento['logicalCores']} threads), {equipamento['system']}, Python {equipamento['python']}, "
            f"ONNX Runtime {equipamento['onnxruntime']}, somente CPU. Sem GPU. Os tempos vêm de {foto['runs']} a {unitario['runs']} execuções, depois de aquecimento."
        )
    )
    historias.append(
        montarTabela(
            ["Cenário", "Resolução", "Latência média", "p95", "Taxa"],
            [
                ["Uma imagem pelo modelo", unitario["resolution"], f"{numero(unitario['meanMs'])} ms", f"{numero(unitario['p95Ms'])} ms", f"{numero(unitario['perSecond'], 0)} imagens/s"],
                ["Foto inteira + 9 blocos (10 imagens em lote)", lote["resolution"], f"{numero(lote['meanMs'])} ms", f"{numero(lote['p95Ms'])} ms", f"{numero(lote['perSecond'], 0)} lotes/s"],
                ["Pipeline completo de uma foto (recorte, pré-processamento e inferência)", foto["resolution"], f"{numero(foto['meanMs'])} ms", f"{numero(foto['p95Ms'])} ms", f"{numero(foto['perSecond'], 1)} fotos/s"],
                ["Vídeo pela API (quadros reduzidos para até 1024 px)", "640 x 480", f"{numero(exemploVideo.averageLatencyMs, 1)} ms por quadro", "-", f"{numero(exemploVideo.processingFps, 1)} FPS"],
            ],
            [5.6 * cm, 2.3 * cm, 3.4 * cm, 1.8 * cm, 3.9 * cm],
        )
    )
    historias.append(Spacer(1, 6))
    historias.append(
        paragrafo(
            "O FPS do vídeo é a taxa de quadros <b>analisados</b> por segundo de inferência (grade 3 x 3 incluída), sem contar leitura e codificação do arquivo. Como analisamos 1 quadro a cada 5, "
            "a taxa efetiva de um vídeo em tempo real pode chegar a cerca de 5 vezes essa taxa de análise, na mesma máquina."
        )
    )

    historias.append(Paragraph("8. Escolhas de pré-processamento, arquitetura e limiar", estiloSecao))
    historias.append(marcador("<b>Qualidade contra velocidade:</b> o modelo leve responde em milissegundos, então sobra margem para analisar nove blocos por foto. Um modelo maior ganharia pouco nesta base, que já está saturada."))
    historias.append(marcador(f"<b>Limiar {numero(limiar)} em vez de 0,50:</b> no teste, as duas escolhas ficam praticamente empatadas. Mantivemos o limiar escolhido na validação (menor limiar com F2 máximo), e não o do resultado no teste."))
    historias.append(marcador("<b>Efeito do limiar:</b> limiar mais baixo sinaliza mais locais e gera mais alarmes falsos, mais alto faz o contrário. Por isso o usuário pode ajustá-lo na tela."))
    historias.append(marcador("<b>Decisão por foto:</b> a foto é sinalizada se a foto inteira ou qualquer bloco passa do limiar. É uma regra conservadora, que aumenta alarmes falsos em fotos grandes."))

    historias.append(Paragraph("9. Limitações e ética", estiloSecao))
    historias.append(marcador("Treinado e testado só em concreto de laboratório. Em fachadas reais o desempenho pode ser bem pior, e isso não foi medido."))
    historias.append(marcador("Detecta apenas indício de fissura. Não mede abertura, profundidade, causa nem risco, e não detecta infiltração, armadura exposta ou buracos."))
    historias.append(marcador("O agrupamento por aparência é uma aproximação, já que a base não informa a imagem de origem de cada recorte. Pode haver recortes da mesma origem em conjuntos diferentes."))
    historias.append(marcador("Foto de pessoas ou de placas pode aparecer na imagem. Por isso nada é gravado no servidor, e a conta pode ser apagada."))
    historias.append(marcador("Erros de rótulo da base afetam as métricas de avaliação."))
    historias.append(marcador("ODS 3 (saúde), ODS 9 (infraestrutura) e ODS 11 (cidades sustentáveis): apoio à manutenção preventiva de unidades de saúde. A contribuição é a de um protótipo."))

    historias.append(Paragraph("10. Como reproduzir", estiloSecao))
    historias.append(paragrafo("Os comandos estão no README do repositório e no notebook <i>vision/notebooks/crackClassification.ipynb</i>: <i>prepareData</i>, <i>trainModel</i>, <i>evaluateModel</i>, <i>benchmarkModel</i> e <i>buildReport</i>. A semente aleatória é fixa em 42."))
    return historias


def executar():
    metricas = json.loads(settings.caminhoMetricas.read_text(encoding="utf-8"))
    benchmark = json.loads(settings.caminhoBenchmark.read_text(encoding="utf-8"))
    pastaImagens = settings.raizProjeto / "samples" / "imagens"
    pastaVideos = settings.raizProjeto / "samples" / "videos"

    exemploImagem = analisarImagem((pastaImagens / "paredeComFissura.jpg").read_bytes(), None, classificarLote)
    exemploVideo = analisarVideo((pastaVideos / "varreduraDaFachada.mp4").read_bytes(), None, classificarLote)

    settings.pastaRelatorio.mkdir(parents=True, exist_ok=True)
    documento = SimpleDocTemplate(
        str(caminhoRelatorio),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=2 * cm,
        title="Inspeção predial com visão computacional",
        author="CitizenTech",
    )
    documento.build(montarHistorias(metricas, benchmark, exemploImagem, exemploVideo), onFirstPage=rodape, onLaterPages=rodape)
    print(f"Relatório salvo em {caminhoRelatorio}")


executar()
