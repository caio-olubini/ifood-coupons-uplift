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

### REQ-213 — Calibração da magnitude do uplift
WHEN o modelo de uplift é avaliado, the system SHALL medir não só a **ordenação** (REQ-203)
mas a **calibração da magnitude**: agrupar o holdout em bins por τ previsto e, em cada bin,
comparar o uplift **previsto** médio contra o uplift **observado** — a diferença entre a
taxa de conversão dos tratados e a dos controles *dentro do bin*.

Qini pode ser alto com magnitude errada: um modelo que prevê 40 p.p. onde o efeito real é
4 p.p. ordena perfeitamente e ainda assim mente sobre o tamanho — e a política de REQ-204
multiplica esse número por receita, então erro de magnitude vira erro de R$. A calibração
é o que separa "ordena bem" de "acerta o quanto".

O uplift observado num bin exige tratados **e** controles no bin (é uma diferença de duas
taxas). Um bin sem um dos braços não tem uplift observável e SHALL ser marcado inavaliável,
não reportado como zero — mesmo princípio de positividade da Premissa 8 aplicado por bin.

Acceptance:
- GIVEN o holdout com τ previsto THEN há, por bin, o uplift previsto médio e o observado
  (tratado − controle), mais um resumo do erro de calibração (ex.: MAE previsto vs. observado).
- GIVEN um bin sem tratado ou sem controle THEN seu uplift observado é marcado inavaliável,
  não zero.

### REQ-214 — Calibração isotônica pós-hoc
WHEN a magnitude do uplift (REQ-213) está mal calibrada, the system SHALL ajustar uma
correção isotônica (`sklearn.isotonic.IsotonicRegression`) que mapeia τ previsto → τ
calibrado, monotônica por construção (preserva a ordenação de REQ-203, corrige só a
magnitude), e reportar as estatísticas de calibração (MAE, bias, por bin) **antes e
depois** da correção, lado a lado.

O ajuste usa **cross-fitting dentro do holdout** (`cfg.calibration_n_folds`): cada fold é
avaliado com a isotônica ajustada nos outros folds, nunca na sua própria fatia — senão o
"depois" aprende no mesmo dado que reporta e fica otimista por construção. Não há um
terceiro split dedicado (o holdout já é a fatia mais recente do dado); cross-fitting reusa
100% do holdout para o "depois" sem violar a disciplina de nunca avaliar no que foi usado
para ajustar.

Acceptance:
- GIVEN o uplift bruto e o calibrado THEN há uma tabela ou par de resumos (MAE, bias) para
  os dois, no mesmo holdout, permitindo comparação direta.
- GIVEN dois pontos calibrados pela **mesma** isotônica (mesmo fold) THEN a ordem de seus
  τ previstos nunca é invertida — a garantia de monotonicidade é por fold, não global:
  cross-fitting usa uma isotônica distinta por fold (ajustada nos outros folds), então a
  ordem entre pontos de folds diferentes pode mudar.
- GIVEN a correção isotônica THEN nenhum ponto do holdout é avaliado pela isotônica
  ajustada com esse mesmo ponto (ou com o bin que o contém).

### REQ-204 — Política sensível a custo
WHEN a alocação é decidida, the system SHALL escolher, por cliente, a oferta (ou "não enviar")
que maximiza `uplift_receita − custo_desconto`, entre as ofertas de **cupom/promoção**
(`offer_type` ∈ {`bogo`, `discount`}) recebidas e a ação nula.

`informational` está **fora de escopo**: não é cupom nem promoção, é uma ação de comunicação
sem desconto associado. Decidir alocação de ações informacionais (quando, para quem, com que
mensagem) é um estudo à parte, não uma extensão desta política — mesmo que o dado tenha
`informational` com custo zero e portanto qualquer uplift positivo o tornasse "lucrativo" por
acidente de contabilidade, não por decisão de produto.

Acceptance:
- GIVEN uma oferta com uplift alto mas custo maior que o ganho THEN não é escolhida.
- GIVEN um sure thing / sleeping dog THEN "não enviar" pode vencer.
- GIVEN uma oferta `informational` entre as opções do cliente THEN nunca é escolhida, mesmo
  com uplift alto e custo zero.

### REQ-205 — Baselines de política
The system SHALL definir três políticas de comparação, restritas ao mesmo escopo de REQ-204
(`bogo`, `discount`): aleatória (oferta uniforme a todos, entre as elegíveis), enviar-a-todos
(status quo que gerou os dados, entre as elegíveis) e top-completion (aloca por probabilidade
prevista de completar, entre as elegíveis).

Acceptance:
- GIVEN o conjunto de avaliação THEN cada baseline produz uma recomendação por cliente com
  pelo menos uma oferta elegível.
- GIVEN uma oferta `informational` entre as opções do cliente THEN nenhum baseline a escolhe.

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
