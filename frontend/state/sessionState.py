import streamlit as st

chavesDaSessao = (
    "token",
    "emailUsuario",
    "registros",
    "idAtual",
    "buscaUpas",
    "inspecao",
    "etapaAuth",
    "emailPendente",
    "confirmarApagar",
)


def iniciarEstado():
    st.session_state.setdefault("token", None)
    st.session_state.setdefault("emailUsuario", None)
    st.session_state.setdefault("registros", [])
    st.session_state.setdefault("idAtual", None)
    st.session_state.setdefault("buscaUpas", None)
    st.session_state.setdefault("inspecao", {"imagem": None, "video": None})
    st.session_state.setdefault("etapaAuth", "entrar")
    st.session_state.setdefault("emailPendente", None)


def estaLogado():
    return st.session_state["token"] is not None


def obterToken():
    return st.session_state["token"]


def obterEmailUsuario():
    return st.session_state["emailUsuario"]


def definirSessao(token, email, registros):
    st.session_state["token"] = token
    st.session_state["emailUsuario"] = email
    st.session_state["registros"] = list(registros)
    st.session_state["idAtual"] = None
    st.session_state["etapaAuth"] = "entrar"
    st.session_state["emailPendente"] = None


def encerrarSessao():
    for chave in chavesDaSessao:
        st.session_state.pop(chave, None)
    iniciarEstado()


def obterEtapaAuth():
    return st.session_state["etapaAuth"]


def obterEmailPendente():
    return st.session_state["emailPendente"]


def definirEtapaAuth(etapa, emailPendente=None):
    st.session_state["etapaAuth"] = etapa
    st.session_state["emailPendente"] = emailPendente


def definirBuscaUpas(resultado):
    st.session_state["buscaUpas"] = resultado


def obterBuscaUpas():
    return st.session_state["buscaUpas"]


def obterRegistros():
    return st.session_state["registros"]


def adicionarRegistro(registro):
    st.session_state["registros"].insert(0, {**registro, "review": None})
    st.session_state["idAtual"] = registro["recordId"]


def obterRegistroAtual():
    idAtual = st.session_state["idAtual"]
    for registro in st.session_state["registros"]:
        if registro["recordId"] == idAtual:
            return registro
    return None


def aplicarRevisao(idRegistro, revisao):
    for registro in st.session_state["registros"]:
        if registro["recordId"] == idRegistro:
            registro["review"] = revisao


def limparAtual():
    st.session_state["idAtual"] = None


def definirInspecao(tipo, dados):
    st.session_state["inspecao"][tipo] = dados


def obterInspecao(tipo):
    return st.session_state["inspecao"][tipo]
