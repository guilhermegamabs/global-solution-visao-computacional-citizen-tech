# Orienta Saúde: Inspeção Predial com Visão Computacional

Entrega de Visão Computacional da Global Solution 2026/2 (CitizenTech). Estende o protótipo Orienta Saúde com uma aba **Inspeção predial**: o usuário envia uma foto ou um vídeo curto de uma UBS ou UPA, e o app sinaliza possíveis fissuras no concreto, mostrando a região suspeita e a confiança do modelo.

**Professor:** Ícaro Gonçalves

**Integrante:** Guilherme Gama Bitencourt Souza (RM565293)

## Problema e limite de uso

Unidades de saúde precisam de manutenção predial para funcionar com segurança, e a prefeitura tem poucas equipes para inspecionar. O módulo ajuda a **priorizar visitas**: ele aponta onde há indício de fissura, não mede o dano.

> A presença de fissura na imagem, isoladamente, não permite determinar a gravidade do dano nem a segurança estrutural da edificação. A decisão é sempre de um profissional habilitado.

## Solução

| Item | Escolha |
|---|---|
| Tarefa | Classificação binária: com e sem fissura |
| Base | Concrete Crack Images for Classification (Mendeley), amostra balanceada |
| Modelo | MobileNetV3-Small pré-treinada no ImageNet, última camada trocada por 2 classes |
| Treino | Fase 1 só a cabeça de classificação, fase 2 ajuste fino de toda a rede |
| Implantação | ONNX Runtime na CPU, sem PyTorch no servidor |
| Visual | Foto dividida em grade 3×3; blocos acima do limiar são marcados em vermelho |
| Vídeo | 1 quadro a cada 5, até 90 quadros, com linha do tempo e quadros mais suspeitos |

### Por que esta arquitetura

- **MobileNetV3-Small:** tem cerca de 2,5 milhões de parâmetros e roda em tempo real na CPU, que é o que um servidor gratuito oferece. Com transferência de aprendizado, poucos milhares de imagens bastam.
- **Entrada 224×224:** é a resolução nativa da rede pré-treinada. As imagens da base têm 227×227, então quase não há perda.
- **Normalização do ImageNet:** necessária porque os pesos foram treinados com ela. O mesmo cálculo está no treino, na avaliação e no backend.
- **Aumento de dados:** giros, rotação, brilho, contraste e desfoque leve, para tolerar sombra e iluminação diferentes.
- **Grade 3×3:** a base é feita de recortes pequenos de superfície, enquanto fotos reais são maiores. Classificar blocos usa o modelo na escala em que ele foi treinado e mostra onde está a suspeita.
- **Limiar de confiança:** escolhido na validação pelo maior F2, que pesa mais o recall. Na triagem, deixar passar uma fissura custa mais do que mandar verificar um local sem dano. O usuário pode ajustar o limiar na tela. Limiar baixo sinaliza mais locais e gera mais alarmes falsos, limiar alto faz o contrário.

## Estrutura

```
backend/app/
  routers/inspectionRouter.py       rotas /inspection
  pipelines/inspectionPipeline.py   grade, decisão, sobreposição, vídeo
  providers/classifierProvider.py   ONNX Runtime e pré-processamento
  schemas/inspectionSchemas.py      modelos de entrada e saída
frontend/features/inspectionScreen.py   aba Inspeção predial
vision/
  settings.py                       caminhos e parâmetros
  scripts/                          prepareData, trainModel, evaluateModel, benchmarkModel, buildSamples, buildReport
  notebooks/crackClassification.ipynb
  data/                             imagens (fora do Git)
models/                             mobilenetCrack.onnx e metrics.json
docs/prints, docs/relatorio         figuras, benchmark e relatório
samples/                            imagens e vídeos de demonstração
```

## Como rodar

Use um ambiente virtual em caminho curto (o torch falha no Windows com caminhos longos), por exemplo `C:\vcvenv`.

```bash
python -m venv C:\vcvenv
C:\vcvenv\Scripts\python.exe -m pip install -r backend/requirements.txt -r vision/requirements.txt -r frontend/requirements.txt
```

### 1. Dados e modelo

Baixe a base em https://data.mendeley.com/datasets/5y9wdsg2zt/1 e extraia de modo que existam `vision/data/raw/Positive` e `vision/data/raw/Negative`. Depois, na raiz do repositório:

```bash
python -m vision.scripts.prepareData
python -m vision.scripts.trainModel
python -m vision.scripts.evaluateModel
python -m vision.scripts.benchmarkModel
```

Variáveis opcionais: `SAMPLEPERCLASS` (padrão 3000), `CLUSTERS` (padrão 150), `HEADEPOCHS`, `TUNEEPOCHS`, `PRETRAINED`, `VISIONDATA`. Os mesmos passos estão no notebook `vision/notebooks/crackClassification.ipynb`.

Para gerar os exemplos de demonstração e o relatório em PDF:

```bash
python -m vision.scripts.buildSamples
python -m vision.scripts.buildReport
```

Os exemplos usam recortes do conjunto de teste montados em mosaico, não fotos reais de unidades de saúde.

O treino gera `models/mobilenetCrack.onnx` e a avaliação gera `models/metrics.json`, as figuras em `docs/prints/` e a medição de desempenho em `docs/relatorio/benchmark.json`.

### 2. Aplicação

Copie `.env.example` para `.env` e preencha as variáveis. Depois:

```bash
cd backend
uvicorn app.main:aplicacao --port 8000
```

```bash
cd frontend
streamlit run app.py
```

Para testar sem e-mail real, use `MOSTRARCODIGONOLOG=true`. As rotas novas estão em `http://localhost:8000/docs`, na seção Inspeção predial.

### 3. Testes

```bash
cd backend
pytest
```

Os testes de inspeção usam um classificador falso e não dependem do modelo treinado.

## Validação do conjunto de dados

A base é formada por recortes de 458 imagens grandes, e os arquivos (`00001.jpg`, `00001_1.jpg`) têm um número único cada, então o nome **não** diz de qual foto o recorte saiu. A numeração também não ajuda: nas imagens com fissura, vizinhas na numeração são parecidas, mas nas imagens sem fissura não há relação.

Por isso o `prepareData` agrupa as imagens por aparência (miniatura 16×16 em cor, PCA e KMeans com 150 grupos) e divide com `StratifiedGroupKFold`, sem grupo em mais de um conjunto (o script confere e imprime 0 grupos compartilhados). O teste mede superfícies que o modelo não viu.

**Limite:** o agrupamento é uma aproximação da origem, então recortes da mesma foto grande podem ter caído em conjuntos diferentes e as métricas podem estar otimistas.

## Resultados

Amostra de 6000 imagens (4292 treino, 853 validação, 855 teste). O limiar 0,40 foi escolhido na validação (menor limiar com F2 máximo). O relatório completo, com figuras e análise de erros, está em [`docs/relatorio/relatorioInspecaoPredial.pdf`](docs/relatorio/relatorioInspecaoPredial.pdf).

| Teste (855 imagens) | Precisão | Recall | F1 | Acurácia | VN / FP / FN / VP |
|---|---|---|---|---|---|
| Limiar 0,40 (escolhido) | 99,31% | 100,00% | 99,65% | 99,65% | 420 / 3 / 0 / 432 |
| Limiar 0,50 | 99,54% | 100,00% | 99,77% | 99,77% | 421 / 2 / 0 / 432 |

**Estes números não representam fachadas reais.** A base é fácil (fissuras nítidas, concreto de laboratório) e o F1 de validação chegou a 1,0. Não medimos fotos de campo.

Robustez a perturbações sintéticas nas imagens de teste, com o mesmo limiar:

| Perturbação | Precisão | Recall | F1 |
|---|---|---|---|
| Original | 99,31% | 100,00% | 99,65% |
| Imagem escura | 99,08% | 100,00% | 99,54% |
| Imagem clara | 99,76% | 97,45% | 98,59% |
| Sombra lateral | 99,08% | 100,00% | 99,54% |
| Baixo contraste | 99,54% | 99,77% | 99,65% |
| Desfoque | 96,86% | 100,00% | 98,41% |
| Ruído | 99,76% | 96,06% | 97,88% |

**Erros:** 3 falsos positivos e 0 falsos negativos. Pela nossa leitura visual, dois dos falsos positivos têm marcas que parecem fissuras, então parte do erro pode ser ruído de rótulo da base. Sem confirmação por especialista.

### Desempenho

Equipamento: AMD Ryzen 5 5600 6-Core Processor (12 threads), Windows 11, ONNX Runtime 1.31.1, somente CPU.

| Cenário | Resolução | Latência média | p95 | Taxa |
|---|---|---|---|---|
| Uma imagem pelo modelo | 224x224 | 0,82 ms | 1,02 ms | 1215 imagens/s |
| Foto inteira + 9 blocos (lote de 10) | 224x224 | 5,23 ms | 6,18 ms | 191 lotes/s |
| Pipeline completo de uma foto | 1024x768 | 26,05 ms | 28,84 ms | 38,4 fotos/s |

No vídeo de demonstração (`samples/videos/varreduraDaFachada.mp4`, 640×480, 10 s), a API analisou 30 quadros a cerca de 23 ms por quadro e 42 FPS de análise.

## Ética, privacidade e limites

- O módulo não diagnostica nem avalia segurança estrutural. O aviso aparece em todo resultado.
- Fotos e vídeos são processados em memória e **não são gravados** no servidor nem no banco.
- É exigido login, como no restante do Orienta Saúde.
- A base é de concreto de laboratório. Fachadas reais, pintura, azulejo, manchas, sombras e vegetação podem gerar falsos positivos ou falsos negativos.
- Não detecta outros danos, como infiltração, armadura exposta ou buracos.
- O servidor processa no máximo 10 MB por foto e 60 MB por vídeo.

## ODS

- **ODS 3, Saúde e Bem-Estar:** unidades de saúde em bom estado de conservação sustentam o atendimento.
- **ODS 9, Indústria, Inovação e Infraestrutura:** apoia a manutenção preventiva de infraestrutura pública.
- **ODS 11, Cidades e Comunidades Sustentáveis:** ajuda a priorizar a inspeção de edificações públicas.
