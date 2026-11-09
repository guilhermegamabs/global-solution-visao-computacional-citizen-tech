from fastapi import Depends, Header

from app.errors import AuthRequiredError
from app.pipelines.authPipeline import obterUsuarioPorToken
from app.providers.classifierProvider import classificarLote
from app.providers.cepProvider import buscarCoordenadasPorCep, buscarEnderecoPorCep
from app.providers.cnesProvider import listarUpasDoMunicipio
from app.providers.emailProvider import enviarCodigoPorEmail
from app.providers.geminiProvider import gerarEstruturado
from app.providers.geocodeProvider import buscarPontoPorNome, geocodificar


def obterGerador():
    return gerarEstruturado


def obterEnviadorEmail():
    return enviarCodigoPorEmail


def obterFontesLocalizacao():
    return {
        "buscarCep": buscarEnderecoPorCep,
        "geocodificar": geocodificar,
        "listarUnidades": listarUpasDoMunicipio,
        "coordenadasPorCep": buscarCoordenadasPorCep,
        "pontoPorNome": buscarPontoPorNome,
    }


def obterTokenBearer(authorization: str | None = Header(default=None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise AuthRequiredError()
    return authorization[len("Bearer "):]


def obterUsuarioAtual(token: str = Depends(obterTokenBearer)):
    usuario = obterUsuarioPorToken(token)
    if not usuario:
        raise AuthRequiredError()
    return usuario


def obterClassificador():
    return classificarLote
