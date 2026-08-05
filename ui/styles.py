from __future__ import annotations

import html

import streamlit as st


CSS = """
<style>
:root {
  --brand:#1F3A5F;
  --brand2:#315B8A;
  --accent:#0F766E;
  --surface:#F7F9FC;
  --danger:#B42318;
  --warning:#9A6700;
}
.block-container {max-width: 1380px; padding-top: 1.5rem; padding-bottom: 3rem;}
[data-testid="stSidebar"] {background:#F4F7FB; border-right:1px solid #E3E8EF;}
.hero {padding:1.35rem 1.5rem; border:1px solid #DDE5EE; border-radius:18px;
       background:linear-gradient(135deg,#F8FBFF 0%,#EEF5FC 100%); margin-bottom:1rem;}
.hero h1 {font-size:2rem; color:#172B4D; margin:0 0 .35rem 0;}
.hero p {color:#526071; margin:0; max-width:900px;}
.hero-meta {margin-top:.8rem; display:flex; gap:.4rem; flex-wrap:wrap;}
.card {background:white; border:1px solid #E3E8EF; border-radius:16px; padding:1rem 1.1rem;
       box-shadow:0 2px 8px rgba(16,24,40,.04); margin-bottom:.8rem;}
.stepper {display:flex; gap:.45rem; flex-wrap:wrap; margin:.7rem 0 1.2rem;}
.step {padding:.45rem .8rem; border-radius:999px; font-size:.86rem; border:1px solid #DDE5EE;
       color:#526071; background:#fff;}
.step.active {background:#E6F0FA; color:#173B63; border-color:#6D9ECC; font-weight:700;
              box-shadow:0 0 0 2px rgba(49,91,138,.08);}
.step.done {background:#E8F7F2; color:#0B5D52; border-color:#91D2C3;}
.badge {display:inline-block; padding:.2rem .55rem; border-radius:999px; font-size:.78rem; font-weight:650;}
.badge-info {background:#E8F1FB;color:#24527A}.badge-ok {background:#E7F6EE;color:#16633B}
.badge-neutral {background:#F1F3F6;color:#526071}
.badge-warn {background:#FFF4D8;color:#7A5200}.badge-danger {background:#FDECEC;color:#9B2C2C}
.status-badge {display:inline-block; padding:.32rem .7rem; border-radius:999px; font-size:.82rem;
               font-weight:700; white-space:nowrap;}
.status-info {background:#E8F1FB;color:#24527A}.status-ok {background:#E7F6EE;color:#16633B}
.status-warn {background:#FFF4D8;color:#7A5200}.status-danger {background:#FDECEC;color:#9B2C2C}
.priority-badge {display:inline-block; min-width:2.2rem; text-align:center; padding:.25rem .45rem;
                 border-radius:999px; font-size:.78rem; font-weight:750;}
.priority-1,.priority-2 {background:#E7F0FA;color:#173B63}.priority-3 {background:#FFF4D8;color:#7A5200}
.priority-4,.priority-5 {background:#F1F3F6;color:#526071}
.recommendation-card {display:flex; align-items:center; justify-content:space-between; gap:1rem;
                      padding:.9rem 1rem; border:1px solid #DDE5EE; border-radius:12px;
                      background:#F8FBFF; margin:.35rem 0 1rem;}
.recommendation-card span {font-size:.83rem;color:#697586}.recommendation-card strong {color:#172B4D;font-size:1.2rem}
.small-muted {font-size:.83rem;color:#697586}.source {font-size:.8rem;color:#526071}
.stButton > button, .stFormSubmitButton > button {border-radius:10px; font-weight:650; min-height:2.7rem;}
button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {
  background:#315B8A !important; border-color:#315B8A !important; color:#fff !important;
}
button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
  background:#244A74 !important; border-color:#244A74 !important;
}
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {
  border:1px solid #E3E8EF; border-radius:12px; overflow:hidden;
}
[data-testid="stSidebar"] [role="radiogroup"] label {padding:.15rem 0;}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {
  color:#1F3A5F !important;font-weight:700 !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) [data-testid="stMarkdownContainer"] p {
  color:#1F3A5F !important;
}
[data-testid="stSidebar"] [role="radio"][aria-checked="true"] > div:first-child,
[data-testid="stSidebar"] [data-baseweb="radio"] input:checked + div {
  background-color:#315B8A !important; border-color:#315B8A !important;
}
[data-testid="stMetricValue"] {font-size:1.35rem;}
@media (max-width: 900px) {
  .block-container {padding-left:1rem; padding-right:1rem;}
  .hero h1 {font-size:1.55rem;}
}
</style>
"""


def apply_styles() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def render_header(
    provider: str,
    model: str,
    retrieval: str,
    app_version: str | None = None,
    environment: str | None = None,
) -> None:
    provider_safe = html.escape(provider)
    model_safe = html.escape(model)
    retrieval_safe = html.escape(retrieval)
    version_safe = html.escape(app_version or "—")
    environment_safe = html.escape(environment or "—")
    st.markdown(
        f"""
        <div class="hero">
          <h1>PROSPECT-LLM</h1>
          <p>Preparação consultiva B2B com evidências, hipóteses verificáveis, perguntas diagnósticas e validação humana registrada.</p>
          <div class="hero-meta">
            <span class="badge badge-info">LLM: {provider_safe} / {model_safe}</span>
            <span class="badge badge-ok">RAG: {retrieval_safe}</span>
            <span class="badge badge-neutral">Versão: {version_safe}</span>
            <span class="badge badge-neutral">Ambiente: {environment_safe}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stepper(current: int, completed: set[int] | None = None) -> None:
    completed = completed or set()
    labels = ["Base", "Contexto", "Análise", "Validação"]
    html_parts = ['<div class="stepper">']
    for index, label in enumerate(labels, 1):
        css = "active" if index == current else "done" if index in completed else ""
        html_parts.append(f'<span class="step {css}">{index}. {label}</span>')
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)
