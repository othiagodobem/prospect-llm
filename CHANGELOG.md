# Changelog

## 3.2.0 — 31/07/2026

- Substituído `use_container_width=True` por `width="stretch"` em todos os componentes Streamlit.
- Eliminados os avisos de depreciação em tabelas, editores, botões e downloads.
- Atualizado o requisito mínimo do Streamlit para `1.50`.
- Mantido o mesmo comportamento visual: elementos continuam ocupando a largura disponível.

## 3.0.0

- Bloqueio e quarentena de documentos de demonstração/gabarito.
- Gestão de fontes: ativar, desativar e excluir.
- Uso na geração separado da presença no inventário.
- Perfil contextual ampliado.
- Linhagem de revisões entre execuções.
- Snapshot de fontes por execução.
- Fundamentação do grau de confiança das hipóteses.
- Prioridades de perguntas normalizadas e validadas.
- Protocolo de validação ajustado às decisões previstas no artigo.
- Teste de conexão com DeepSeek.
- Exportação do roteiro aprovado em JSON, CSV e DOCX.
- Correção da sincronização entre navegação e stepper.
- Correção do resumo da execução para usar o perfil contextual como fonte de verdade.

## 3.1.0
- Corrigido o ponto de entrada do Streamlit: `render_app()` passa a ser executado ao carregar `app.py`.
- Corrigida a tela em branco que mostrava apenas o menu "Deploy".
