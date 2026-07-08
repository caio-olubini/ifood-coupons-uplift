# Spec: Modelagem, uplift & avaliação offline

> Status: Ready
> Consome: schema-processed.md · Implementa a fundação (00-clarify.md).

## Purpose

A partir do dataset unificado, estimar o efeito incremental de cada oferta por cliente,
decidir a alocação que maximiza lucro líquido (incluindo não enviar), e provar o ganho da
política sem A/B, avaliando-a offline contra baselines de campanha realistas.

## Scope

**In scope**
- Baseline preditivo (logística → LGBM) para validar que há sinal aprendível.
- Modelo de uplift X-learner com contrato de saída (uplift por cliente × oferta).
- Política de alocação multi-tratamento sensível a custo, com a ação "não enviar".
- Três baselines de política: aleatória, enviar-a-todos, top-completion.
- Avaliação offline por IPW (propensity fixa conhecida) da política vs. baselines.
- Métricas de uplift (Qini/AUUC) e impacto em R$.
- Tracking de experimentos em MLflow.

**Out of scope**
- Pipeline, label, features (spec 01).
- A/B online (próximo passo dimensionado).

**Non-goals**
- Não selecionar modelo por AUC/F1. Uplift exige métrica de uplift.
- Não estimar propensity para de-confounding (RCT, propensity conhecida — Premissa 4).
- Não prometer valor absoluto para "não enviar a ninguém" via IPW (positividade — Premissa 8).

## Users & stories

- Como cientista de dados, quero uplift por cliente × oferta num contrato estável, para a
  política decidir alocação sem conhecer o estimador.
- Como líder de negócio, quero o ganho em R$ contra o status quo, para decidir investir.
- Como revisor, quero a política avaliada contra baselines realistas, para confiar no número
  antes de qualquer gasto.

## Functional requirements

### REQ-201 — Baseline preditivo
WHEN o dataset é modelado, the system SHALL treinar uma regressão logística como âncora e um
LGBM sob validação temporal, registrando ambos.

Acceptance:
- GIVEN validação temporal (nunca split aleatório) THEN o LGBM iguala ou supera a logística;
  caso contrário há bug a investigar.

### REQ-202 — Uplift X-learner
WHEN o uplift é estimado, the system SHALL usar um X-learner treinado com `treatment`
(exposição = viu a oferta) e `converted`, produzindo uplift por cliente × tipo de oferta.

Acceptance:
- GIVEN um cliente sure thing (μ₀ já alto) THEN o uplift estimado tende a zero.
- GIVEN a saída THEN há uma estimativa de uplift por par cliente × tipo de oferta.

### REQ-203 — Métrica de uplift
WHEN um modelo de uplift é avaliado, the system SHALL usar Qini/AUUC, não métricas de
classificação, para seleção e comparação.

Acceptance:
- GIVEN dois modelos de uplift THEN a escolha se dá por Qini/AUUC reportado.

### REQ-204 — Política sensível a custo
WHEN a alocação é decidida, the system SHALL escolher, por cliente, a oferta (ou "não enviar")
que maximiza `uplift_receita − custo_desconto`, entre todas as ofertas e a ação nula.

Acceptance:
- GIVEN uma oferta com uplift alto mas custo maior que o ganho THEN não é escolhida.
- GIVEN um sure thing / sleeping dog THEN "não enviar" pode vencer.

### REQ-205 — Baselines de política
The system SHALL definir três políticas de comparação: aleatória (oferta uniforme a todos),
enviar-a-todos (status quo que gerou os dados) e top-completion (aloca por probabilidade
prevista de completar).

Acceptance:
- GIVEN o conjunto de avaliação THEN cada baseline produz uma recomendação por cliente.

### REQ-206 — Avaliação offline por IPW
WHEN uma política é avaliada offline, the system SHALL estimar seu valor por IPW usando a
propensity fixa conhecida do experimento, sobre o conjunto retido.

Acceptance:
- GIVEN a política de uplift e os três baselines THEN há um valor estimado (lucro líquido) para
  cada, na mesma base retida.
- GIVEN a política de uplift THEN seu valor estimado supera os três baselines, ou a diferença é
  reportada honestamente.

### REQ-207 — Limite de positividade
IF uma política recomenda uma ação sem sobreposição no observado (ex.: não enviar a ninguém),
THEN the system SHALL marcar seu valor como não estimável por IPW puro, em vez de reportar um número.

Acceptance:
- GIVEN "não enviar a ninguém" THEN o relatório declara a inavaliabilidade, não um valor.

### REQ-208 — Impacto em R$
WHEN o resultado é comunicado, the system SHALL expressar o ganho em R$ (economia de desconto
e/ou receita incremental) com intervalo, liderando pela economia de desconto.

Acceptance:
- GIVEN a política vencedora THEN há um valor em R$ com intervalo, derivado da avaliação offline.

### REQ-209 — Tracking de experimentos
The system SHALL registrar em MLflow, para cada run, parâmetros, métricas (incl. Qini/AUUC e
lucro por política) e artefatos (SHAP, tabela de políticas), com nomes de run estáveis.

Acceptance:
- GIVEN qualquer modelo treinado THEN existe um run MLflow com params, métricas e artefatos.

### REQ-210 — Configurabilidade
The system SHALL carregar todo parâmetro de modelagem (custos por tipo de oferta, corte
temporal de validação, hiperparâmetros, seeds) do objeto de config validado, sem hardcode.

Acceptance:
- GIVEN o código de modelagem THEN nenhum custo/corte/hiperparâmetro aparece literal fora da config.

### REQ-211 — Próximo passo dimensionado
WHEN o relatório propõe A/B, the system SHALL dimensionar N por braço a partir da variância
observada, não propor "rodar A/B" genérico.

Acceptance:
- GIVEN a proposta de A/B THEN há um N por braço com a variância que o originou.

## Non-functional requirements

- **Validação temporal:** todo treino/avaliação respeita a ordem temporal e as ondas de
  campanha; nenhum split aleatório; nenhum cliente vaza entre treino e teste.
- **Reprodutibilidade:** modelagem roda por CLI sob UV; seeds na config; runs em MLflow.
- **Visualização:** figuras Plotly no mesmo tema executivo da spec 01 — curva Qini, tabela de
  lucro por política, force plot SHAP individual para o slide. Baixa carga visual.
- **Notebooks apenas para análise:** lógica de modelo/política/avaliação em `src/`; notebooks
  exibem.

## Domain entities

- **Estimativa de uplift**: `account_id`, `offer_type`, `uplift`, `expected_net_profit`.
- **Recomendação de política**: `account_id`, `chosen_action` (oferta ou nenhuma),
  `expected_net_profit` — objeto tipado (Pydantic) antes de virar tabela/MLflow.
- **Resultado de avaliação**: `policy_name`, `estimated_net_profit`, `interval`, `evaluable`.

## Assumptions & open questions

- Premissas 4, 5, 6, 8 de 00-clarify.md aplicam-se diretamente.
- Depende do dataset processado conforme schema-processed.md.
- Sem clarificações pendentes em escopo obrigatório.

## Success criteria

- Uplift por cliente × oferta produzido e avaliado por Qini/AUUC.
- Política de uplift supera os três baselines na avaliação offline por IPW.
- Ganho em R$ com intervalo, liderando pela economia de desconto, comunicável a leigos.
- Próximo passo é um A/B com N por braço calculado da variância observada.
