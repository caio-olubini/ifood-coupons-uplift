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
- Avaliação offline por curva de ganho incremental por budget top-N, contrafactual observado no
  dado real, comparando as estratégias uplift × conversão crua × aleatória.
- Métricas de uplift (Qini/AUUC).
- Tracking de experimentos em MLflow.

**Out of scope**
- Pipeline, label, features (spec 01).
- A/B online e seu dimensionamento (descontinuado com o IPW — ver REQ-211).
- Avaliação por IPW / Direct Method: operam sobre receita bruta realizada, não sobre o
  incremental causal que a curva de ganho isola (ver REQ-206).

**Non-goals**
- Não selecionar modelo por AUC/F1. Uplift exige métrica de uplift.
- Não estimar propensity para de-confounding (RCT, propensity conhecida — Premissa 4).
- Não avaliar sobre receita bruta observada: o objeto é o ganho **incremental** dos top-N de
  cada estratégia, com contrafactual observado.

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

> **Resolvido por decisão de contrato (G3).** O label deixou de exigir view: `converted`
> mede compra na validade atingindo o `min_value`, tenha o cliente visto a oferta ou não.
> Ver é o *tratamento*, não o rótulo. Com isso o controle tem outcome positivo — 33,0%
> (7.140/21.623) contra 49,2% no tratado — e μ₀ > 0 em todos os `offer_type`
> (μ₀ ∈ [0,36; 0,44] no holdout). O X-learner volta a estimar τ = μ₁ − μ₀ de fato.
>
> Antes da correção, `converted=1` implicava view, o controle não tinha nenhum outcome
> positivo, μ₀ ≡ 0 e τ degenerava em μ₁ (uplift "médio" de 45,8 p.p., Qini 0,548 — altos
> **pelo** defeito). Depois: uplift médio de 4,7 p.p. (bogo), 6,4 (discount) e 10,2
> (informational), Qini AUC 0,0385. O número menor é o honesto — e o teste de placebo
> (REQ-212) confirma que não é ruído: a distribuição nula (`treatment` embaralhado dentro
> de cada `offer_type`, 20 réplicas) tem média ≈ 0 e desvio 0,011; o Qini real fura o
> percentil 95 (0,0162) com p-valor empírico 0/20.
> `test_pipeline_label_admits_conversion_in_control` guarda o invariante;
> `test_label_impossible_in_control_degenerates_uplift_into_mu1` fixa a assinatura
> numérica da regressão.
>
> **Confundimento residual, registrado, não resolvido:** `treatment` = viu a oferta
> continua sendo **escolha do cliente**, não braço aleatorizado (o que foi randomizado é o
> *envio*, Premissa 4). Quem abre a oferta já tende a ser mais ativo, e essa auto-seleção
> entra na estimativa. O uplift aqui é causal **sob a premissa de ignorabilidade
> condicional às features** `hist_*`/perfil — não por desenho experimental. Leia o número
> com essa qualificação; a Premissa 8 (positividade) segue valendo para "não enviar a
> ninguém".

### REQ-203 — Métrica de uplift
WHEN um modelo de uplift é avaliado, the system SHALL usar Qini/AUUC, não métricas de
classificação, para seleção e comparação.

Acceptance:
- GIVEN dois modelos de uplift THEN a escolha se dá por Qini/AUUC reportado.

### REQ-212 — Teste de placebo por permutação
WHEN o Qini do modelo de uplift é reportado, the system SHALL testá-lo contra uma
distribuição nula obtida embaralhando `treatment` **dentro de cada `offer_type`** e
**preservando a proporção tratado/controle** do grupo, mantendo `X` (features) e `y`
(`converted`) fixos. O Qini real deve exceder o percentil `cfg.placebo_confidence_level`
da distribuição nula (N réplicas, `cfg.placebo_n_permutations`, sementes distintas).

Um embaralhamento global de `treatment` mudaria a razão tratado/controle por
`offer_type`, derrubando o Qini nulo por um motivo diferente do que o teste quer
isolar (se o modelo aprendeu efeito real, não composição de grupo).

A mesma distribuição nula é o intervalo de confiança do Qini reportado em REQ-203: não
são dois cálculos, é um só lido de duas formas — o percentil que o Qini real precisa
superar (significância) e a dispersão da nula (incerteza do número).

Acceptance:
- GIVEN o Qini real e N réplicas de placebo THEN o Qini real está acima do percentil
  `cfg.placebo_confidence_level` da distribuição nula, ou o relatório declara que o
  modelo não passou no placebo.
- GIVEN uma permutação THEN a proporção tratado/controle de cada `offer_type` é
  idêntica à original.

### ~~REQ-213 — Calibração da magnitude do uplift~~ (descontinuado)
Removido por decisão do usuário (2026-07-10), junto com REQ-204/REQ-205 (política e
baselines de alocação): o diagnóstico de calibração media magnitude para uma política que
não existe mais no projeto. Saíram `uplift_eval.calibration_by_bin`, `calibration_error`,
`fig_calibration`, o campo `cfg.calibration_n_bins` e seus testes.

### ~~REQ-214 — Calibração isotônica pós-hoc~~ (descontinuado)
Removido por decisão do usuário (2026-07-10): a correção isotônica reescalava a **magnitude**
prevista de τ sem tocar a **ordenação** (Qini) — dependia de REQ-213, também descontinuada.

### ~~REQ-204 — Política sensível a custo~~ (descontinuado)
Removido por decisão do usuário (2026-07-10): o projeto passou a focar a avaliação de
modelos de uplift em Qini/AUUC e na curva de ganho incremental por budget (REQ-206), sem uma
etapa de alocação por cliente. `src/policy.py` (`offer_economics`, `expected_net_profit`,
`allocate`) foi removido inteiro, com `tests/test_policy.py`.

### ~~REQ-205 — Baselines de política~~ (descontinuado)
Removido junto com REQ-204 — os três baselines de alocação (`policy_random`,
`policy_send_all`, `policy_top_completion`) não têm mais uma política para comparar. Os
baselines de **estratégia de ranking** (uplift puro, conversão crua, aleatório) continuam
vivos em `uplift_eval.qini_by_strategy`/`gaincurve`, sob REQ-203/REQ-206.

### REQ-206 — Avaliação offline por curva de ganho incremental (budget top-N)
WHEN as estratégias de alocação são avaliadas offline, the system SHALL comparar seu **lucro
líquido incremental por budget top-N** sobre o conjunto retido: cada estratégia é um ranking de
clientes, e para cada budget N a avaliação mede o lucro incremental causal dos top-N daquela
estratégia. O contrafactual é **observado no dado real** (diferença tratado − controle do RCT,
estilo Qini — controle escalado pela razão tratado/controle do prefixo), nunca previsto pelo
modelo. Três estratégias entram: modelo de uplift (ranking por τ previsto), conversão crua
(ranking por P(converte) previsto) e escolha aleatória.

IPW e Direct Method saem de escopo: ambos avaliam sobre **receita bruta realizada**
(`conversion_value − reward_cost` de fato ocorrido), que soma a conversão causada com a que
aconteceria de qualquer forma — não isolam o incremental, que é o objeto desta avaliação.

Acceptance:
- GIVEN as três estratégias THEN há uma curva `(budget N, lucro incremental)` para cada, na mesma
  base retida, e a leitura em budgets destacados (`gain_curve_budgets`).
- GIVEN a estratégia de uplift THEN sua curva domina a aleatória, ou a diferença é reportada
  honestamente.
- GIVEN qualquer prefixo top-N sem controle observado THEN o ganho não desconta contrafactual
  inexistente (não inventa ganho onde não há como estimá-lo).

### ~~REQ-207 — Limite de positividade~~ (descontinuado)
Era o limite de positividade do IPW ("não enviar a ninguém" é inavaliável). Sai com o IPW: a
curva de ganho não estima o valor absoluto de uma ação, e sim o ganho relativo de um ranking
sobre o observado, então não há o buraco de cobertura que REQ-207 protegia.

### ~~REQ-208 — Impacto em R$~~ (absorvido por REQ-206)
O ganho já é expresso em R$ (lucro líquido incremental) pela própria curva de REQ-206. O
intervalo de confiança normal-aproximado e a "economia de desconto em destaque" saíam do
estimador IPW pontual, que não existe mais.

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

### ~~REQ-211 — Próximo passo dimensionado~~ (descontinuado)
Dimensionava N por braço do próximo A/B a partir da variância das contribuições IPW. Sai com o
IPW (a variância que o originava não existe mais) e com a redução do escopo desta entrega ao
comparativo de estratégias por curva de ganho.

## Non-functional requirements

- **Validação temporal:** todo treino/avaliação respeita a ordem temporal e as ondas de
  campanha; nenhum split aleatório; nenhum cliente vaza entre treino e teste.
- **Reprodutibilidade:** modelagem roda por CLI sob UV; seeds na config; runs em MLflow.
- **Visualização:** figuras Plotly no mesmo tema executivo da spec 01 — curva Qini, curva de
  ganho incremental por budget (uma série por estratégia), force plot SHAP individual para o
  slide. Baixa carga visual.
- **Notebooks apenas para análise:** lógica de modelo/política/avaliação em `src/`; notebooks
  exibem.

## Domain entities

- **Estimativa de uplift**: `account_id`, `offer_type`, `uplift`, `expected_net_profit`.
- **Recomendação de política**: `account_id`, `chosen_action` (oferta ou nenhuma),
  `expected_net_profit` — objeto tipado (Pydantic) antes de virar tabela/MLflow.
- **Curva de ganho**: `strategy`, `n` (budget), `gain` (lucro líquido incremental dos top-N).

## Assumptions & open questions

- Premissas 4, 5, 6, 8 de 00-clarify.md aplicam-se diretamente.
- Depende do dataset processado conforme schema-processed.md.
- Sem clarificações pendentes em escopo obrigatório.

## Success criteria

- Uplift por cliente × oferta produzido e avaliado por Qini/AUUC.
- Curva de ganho incremental por budget: a estratégia de uplift entrega mais lucro incremental
  por budget que conversão crua e aleatória, ou a diferença é reportada honestamente.
- Ganho em R$ (lucro líquido incremental) legível por budget, comunicável a leigos.
