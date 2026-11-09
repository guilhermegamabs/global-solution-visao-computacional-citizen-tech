from pathlib import Path

import streamlit as st

from features.aboutScreen import renderizarSobre
from features.authScreen import renderizarAutenticacao
from features.dashboardScreen import renderizarPainel
from features.historyScreen import renderizarHistorico
from features.inspectionScreen import renderizarInspecao
from features.sidebarPanel import renderizarBarraLateral
from features.triageScreen import renderizarTriagem
from features.unitsScreen import renderizarUpas
from state.sessionState import estaLogado, iniciarEstado
from ui.theme import aplicarTema

caminhoIcone = Path(__file__).parent / "assets" / "icone.png"

st.set_page_config(page_title="Orienta Saúde", page_icon=caminhoIcone, layout="wide")

iniciarEstado()
aplicarTema()
renderizarBarraLateral()

st.markdown(
    '<div class="appHeader"><h1>✚ Orienta Saúde</h1>'
    "<p>Descubra onde buscar atendimento, em linguagem simples.</p></div>",
    unsafe_allow_html=True,
)

if not estaLogado():
    renderizarAutenticacao()
    st.stop()

abaTriagem, abaUpas, abaInspecao, abaHistorico, abaPainel, abaSobre = st.tabs(
    ["🩺 Triagem", "🗺️ UPAs próximas", "🏗️ Inspeção predial", "🗂️ Histórico", "📊 Painel", "ℹ️ Sobre"]
)

with abaTriagem:
    renderizarTriagem()
with abaUpas:
    renderizarUpas()
with abaInspecao:
    renderizarInspecao()
with abaHistorico:
    renderizarHistorico()
with abaPainel:
    renderizarPainel()
with abaSobre:
    renderizarSobre()
