# PROSPECT-LLM

**Versão 3.2.0** v3.0

Protótipo funcional do framework sociotécnico para prospecção consultiva B2B em serviços de TI com:

- governança documental;
- base de conhecimento rastreável;
- coleta estruturada de contexto;
- RAG local híbrido;
- geração por DeepSeek em JSON validado;
- matriz de hipóteses;
- roteiro diagnóstico;
- qualificação argumentada;
- validação humana registrada;
- versionamento entre execuções;
- exportação do roteiro aprovado.

## Principais correções da versão 3.0

1. **Proteção contra contaminação da base**
   - guias, gabaritos e documentos com respostas esperadas são bloqueados;
   - materiais antigos suspeitos são colocados em quarentena automaticamente;
   - execuções históricas contaminadas recebem alerta e não podem ser validadas.

2. **Governança de fontes**
   - classificação sem valor pré-selecionado;
   - tipo de fonte;
   - autorização registrada;
   - controle de validade;
   - ativação, desativação e exclusão;
   - flag de participação na geração.

3. **Perfil contextual ampliado**
   - oferta principal;
   - papel decisório separado do cargo;
   - histórico de contatos;
   - restrições conhecidas;
   - fatos com fonte obrigatória por padrão.

4. **Revisões rastreáveis**
   - uma execução corrigida pode ser criada a partir da anterior;
   - o sistema registra execução-pai, número da revisão e motivo.

5. **Melhorias na análise**
   - snapshot das fontes usadas;
   - versão documental nas evidências;
   - fundamentação do grau de confiança;
   - normalização da prioridade das perguntas;
   - alerta para menos de duas perguntas prioritárias.

6. **Validação fiel ao artigo**
   - opções de decisão variam por critério;
   - linguagem comercial e próximo passo permitem apenas aprovar ou ajustar;
   - proteção de dados permite aprovar, anonimizar ou rejeitar;
   - alterações entre perguntas originais e editadas ficam registradas.

7. **UX e operação**
   - stepper sincronizado com a navegação;
   - modelo, versão e ambiente no cabeçalho;
   - teste de conexão com a DeepSeek;
   - exportação do roteiro aprovado em JSON, CSV e DOCX.

## Instalação no Windows

```powershell
cd C:\PROSPECT-LLM\PROSPECT-LLM-DeepSeek-v3
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
python -m pytest -q
python -m streamlit run app.py
```

No `.env`, configure:

```env
DEEPSEEK_API_KEY=sk-sua-chave
```

## Migração da v2

1. Faça backup da pasta anterior e de `data/`.
2. Extraia a v3 em uma pasta nova.
3. Crie um `.env` a partir do novo `.env.example`.
4. Para preservar o histórico, copie apenas `data/prospect_llm.db`, `data/checkpoints.db` e `data/retrieval/`.
5. Na primeira abertura, fontes com nomes de guia ou gabarito serão colocadas em quarentena.
6. Gere uma nova execução para análises que tenham utilizado o guia de demonstração.

As novas tabelas são criadas automaticamente pelo SQLAlchemy. Nenhuma coluna existente foi removida.

## Segurança

- Não compartilhe o `.env`.
- Não envie contratos, credenciais, dados pessoais desnecessários ou valores confidenciais ao MVP em nuvem.
- Documentos restritos ou sensíveis são bloqueados por padrão.
- O documento completo fica no índice local, mas os trechos recuperados são enviados à API DeepSeek.
- Mantenha `ALLOW_RESTRICTED_CLOUD=false`.

## Testes

```powershell
python -m pytest -q
```

Os testes cobrem:

- ingestão e busca local;
- bloqueio de guia/gabarito;
- desativação de fontes;
- normalização de prioridades;
- validação de schemas;
- saída estruturada do provider;
- finalização do workflow.

## Limites do protótipo

- o RAG usa BM25 + TF-IDF, adequado ao MVP e a bases pequenas ou médias;
- não há autenticação corporativa nem controle de acesso por usuário;
- não há validação empírica de impacto comercial;
- não há OCR automático para PDFs digitalizados;
- o protótipo demonstra implementabilidade e rastreabilidade, não aumento de conversão ou receita.
