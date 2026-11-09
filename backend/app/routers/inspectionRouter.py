from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.dependencies import obterClassificador, obterUsuarioAtual
from app.errors import FileTooLargeError
from app.pipelines.inspectionPipeline import analisarImagem, analisarVideo, obterInformacoesModelo
from app.schemas.inspectionSchemas import ImageInspection, ModelInfo, VideoInspection

roteador = APIRouter(prefix="/inspection", tags=["Inspeção predial"])

tamanhoMaximoImagem = 10 * 1024 * 1024
tamanhoMaximoVideo = 60 * 1024 * 1024


def lerArquivo(arquivo, limite):
    conteudo = arquivo.file.read(limite + 1)
    if len(conteudo) > limite:
        raise FileTooLargeError()
    return conteudo


@roteador.get("/model", response_model=ModelInfo)
def consultarModelo(usuario=Depends(obterUsuarioAtual)):
    return obterInformacoesModelo()


@roteador.post("/image", response_model=ImageInspection)
def inspecionarImagem(
    arquivo: UploadFile = File(...),
    limiar: float | None = Form(default=None, alias="threshold", ge=0.05, le=0.99),
    usuario=Depends(obterUsuarioAtual),
    classificar=Depends(obterClassificador),
):
    return analisarImagem(lerArquivo(arquivo, tamanhoMaximoImagem), limiar, classificar)


@roteador.post("/video", response_model=VideoInspection)
def inspecionarVideo(
    arquivo: UploadFile = File(...),
    limiar: float | None = Form(default=None, alias="threshold", ge=0.05, le=0.99),
    usuario=Depends(obterUsuarioAtual),
    classificar=Depends(obterClassificador),
):
    return analisarVideo(lerArquivo(arquivo, tamanhoMaximoVideo), limiar, classificar)
