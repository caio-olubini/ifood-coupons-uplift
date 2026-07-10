# Spec: Pipeline de dados & EDA

> Status: Ready
> Implementa a fundação (00-clarify.md) e produz o contrato (schema-processed.md).

## Purpose

Transformar três arquivos brutos de eventos em um dataset unificado no grão
(cliente, oferta recebida), com label influence-aware e features sem leakage, e produzir
a análise exploratória que permite avaliar os dados e diagnosticar o experimento. Sem esta
base correta, nenhuma modelagem é confiável.

## Scope

**In scope**
- Ingestão e limpeza dos três JSONs em PySpark.
- Parsing do campo `value` com os quatro subcampos reais e a coalescência `offer id`/`offer_id`.
- Label influence-aware e atribuição temporal oferta→transação.
- Features pré-recebimento (transacionais, histórico de resposta, cliente, oferta, contexto).
- Escrita do dataset processado conforme o contrato.
- EDA: distribuição de eventos, ondas de campanha, minas do dataset, check de balanço.

**Out of scope**
- Modelagem, uplift, política, avaliação offline (spec 02).

**Non-goals**
- Não imputar a sentinela de idade para dentro de estatísticas.
- Não usar `offer completed` como sinal de conversão para informational.
- Não validar linha a linha com Pydantic dentro do Spark.

## Users & stories

- Como cientista de dados, quero um dataset no grão certo com garantias testadas, para
  modelar uplift sem reconstruir a lógica de atribuição.
- Como revisor, quero uma EDA legível, para entender os dados e por que o problema é uplift.
- Como responsável pela decisão de modelagem, quero o diagnóstico de balanço, para justificar
  a escolha do estimador.

## Functional requirements

### REQ-101 — Parsing do campo `value`
WHEN uma linha de `transactions` é lida, the system SHALL extrair `amount`, `reward` e uma
referência de oferta única obtida pela coalescência de `offer id` (com espaço) e `offer_id`
(com underscore).

Acceptance:
- GIVEN um evento `offer received` ou `offer viewed` WHEN parseado THEN a referência de oferta
  vem de `offer id` e é não-nula.
- GIVEN um evento `offer completed` WHEN parseado THEN a referência vem de `offer_id` e é não-nula.
- GIVEN um evento `transaction` WHEN parseado THEN `amount` é não-nulo e não há referência de oferta.

### REQ-102 — Normalização do perfil e sentinela de identidade
WHEN o `profile` é carregado, the system SHALL marcar `identity_missing=1` para clientes com
`age=118`, converter `age` sentinela para null, normalizar `gender` ausente para `unknown` e
derivar `tenure_days` a partir de `registered_on`.

Acceptance:
- GIVEN os 2.175 clientes sentinela THEN `identity_missing=1` e `age` é null para todos e só eles.
- GIVEN um cliente com `registered_on=YYYYMMDD` THEN `tenure_days` é inteiro ≥ 0.

### REQ-103 — Atribuição temporal oferta→transação
WHEN uma transação ocorre dentro de `[received_time, received_time + duration]` de uma oferta
vista pelo mesmo cliente **e seu valor atinge o `min_value` da oferta**, the system SHALL atribuir
essa transação àquela oferta, assumindo no máximo uma oferta ativa por cliente no intervalo
(Premissa 1). Uma compra abaixo do gasto mínimo não ativa a recompensa e, portanto, não é
atribuível: contá-la faria o pipeline debitar um desconto que a empresa nunca concederia.

Acceptance:
- GIVEN uma transação fora de qualquer janela de validade THEN não é atribuída a nenhuma oferta.
- GIVEN uma transação com `amount < min_value` da oferta THEN não é atribuída àquela oferta,
  ainda que ocorra pós-view e dentro da validade (garantia G10). Como `informational` tem
  `min_value = 0`, o filtro nunca a alcança.
- GIVEN uma transação disputada por uma oferta inelegível (`amount < min_value`) e outra elegível
  THEN a elegível fica com a transação — o gasto mínimo é filtrado **antes** do desempate de
  posse, para que a inelegível não vença a prioridade e descarte a conversão da elegível.
- GIVEN duas ofertas ativas no mesmo intervalo para um cliente THEN o pipeline aplica a regra de
  prioridade configurada e registra a ocorrência em log de premissa.
- GIVEN um único evento físico (view ou transação) disputado por mais de um recebimento THEN ele
  é atribuído a **um só** recebimento — view e transação são exclusivas, espelhando uma única
  exposição/compra numa única linha do grão.
- GIVEN dois recebimentos com `received_time` idêntico disputando o mesmo evento THEN o desempate
  por `offer_id` torna a atribuição determinística entre execuções (reprodutibilidade).

### REQ-104 — Label influence-aware
IF uma oferta foi vista e uma transação atribuída ocorre **após o view** e dentro da validade,
THEN the system SHALL marcar `converted=1`; caso contrário `converted=0`. Como REQ-103 só atribui
transações que atingem o `min_value`, `converted=1` implica `conversion_value ≥ min_value`.

O limiar é **por transação**, não sobre o gasto acumulado na janela: duas compras de R$ 6 numa
oferta de `min_value = 10` não convertem, porque nenhuma delas sozinha ativou a recompensa.

Acceptance:
- GIVEN uma oferta completada sem view precedente THEN `converted=0`.
- GIVEN uma transação na janela mas **anterior ao view** (comprou e depois viu) THEN `converted=0`
  e a transação não entra em `conversion_value` — é sure thing, não conversão induzida (G4).
- GIVEN uma oferta com transação após `received_time + duration` THEN `converted=0`.
- GIVEN uma transação pós-view e na validade com `amount < min_value` THEN `converted=0` e a
  transação não entra em `conversion_value` (G10).
- GIVEN uma janela com uma transação abaixo e outra acima do `min_value` THEN só a segunda entra
  em `conversion_value`; a primeira é descartada, não somada.
- GIVEN informational vista com transação na janela pós-view THEN `converted=1` sem depender
  de evento `offer completed` (`min_value = 0` não filtra nada).

### REQ-105 — Features sem leakage temporal
WHILE computa qualquer feature `hist_*` para uma linha, the system SHALL usar apenas eventos
com `time < received_time` daquela linha.

Acceptance:
- GIVEN qualquer linha do dataset THEN nenhuma feature incorpora evento em `time ≥ received_time`
  (garantia G2, verificada por teste dedicado).

### REQ-106 — Custo do desconto
WHEN uma conversão de bogo ou discount é registrada, the system SHALL preencher `reward_cost`
com o custo do desconto; para informational e não-convertidos, `reward_cost=0`.

Cobrar o custo em toda conversão só é correto porque REQ-103/G10 garante que a compra atingiu o
`min_value` — o desconto de fato teria sido concedido. Antes de G10 o pipeline debitava desconto
em conversões abaixo do mínimo (25,9% das conversões pagas), inflando o custo e tornando o envio
menos atrativo do que é na função de lucro da spec 02. Fenda fechada na atribuição, não aqui.

Acceptance:
- GIVEN `reward_cost > 0` THEN `converted=1` e `offer_type ≠ informational` (garantia G6).
- GIVEN `reward_cost > 0` THEN `conversion_value ≥ min_value` (garantia G10) — nenhum desconto
  é debitado por uma compra que não o teria disparado.

### REQ-107 — Escrita conforme contrato
WHEN o dataset processado é escrito, the system SHALL impor o `StructType` do contrato e passar
por validação Pydantic de schema e amostra.

Acceptance:
- GIVEN o dataset final THEN todas as garantias G1–G10 do contrato passam.
- GIVEN o dataset final THEN a validação Pydantic de amostra não levanta erro.

### REQ-108 — EDA das decisões do projeto
WHEN a EDA é executada, the system SHALL produzir uma análise **seccionada** que olha o dado e
sustenta as decisões de projeto com números: volumetria, distribuição das features, outliers,
correlação, funil de conversão, segmentos de cliente e diagnóstico do desenho experimental.
Cada seção fecha com uma leitura breve dos números que exibiu.

Cobertura mínima, uma seção cada: (1) panorama e catálogo de ofertas; (2) eventos no tempo com as
ondas marcadas; (3) qualidade do dado — ausência de `offer_id` em transação, completou-sem-ver por
tipo, completou-após-validade, informational sem `offer completed`, sentinela de identidade,
concorrência de ofertas na janela, e a tabela de sanidade do grão; (4) perfil univariado de toda
coluna numérica (nulos, zeros, percentis, cauda por cerca de Tukey) e das categóricas;
(5) correlação com os pares redundantes acima do limiar da config; (6) funil recebido → visto →
convertido e conversão por onda; (7) segmentação de clientes (REQ-111); (8) resposta observada por
segmento; (9) balanço (REQ-109) e positividade por tipo; (10) compra espontânea, que sustenta o
enquadramento de uplift; (11) síntese.

Acceptance:
- GIVEN a EDA THEN cada figura tem título que afirma a conclusão, não o conteúdo, e usa o tema
  de `src/viz.py` (ver requisito não-funcional de visualização).
- GIVEN a EDA THEN o corpo do notebook usa no máximo 12 figuras; análises de suporte sem figura
  própria vão em tabela, não em plot.
- GIVEN a EDA THEN a taxa de completou-sem-ver é reportada por tipo de oferta.
- GIVEN uma taxa de conversão THEN o denominador é explícito — sobre recebidos, sobre vistos, ou
  ambos lado a lado. Taxa sem denominador nomeado é defeito.
- GIVEN a EDA THEN termina numa síntese: tabela decisão → evidência (número computado na
  célula) → onde vive (premissa/REQ) — o índice de justificativa das escolhas técnicas.
- GIVEN uma divergência entre premissa e dado medido THEN ela é registrada com o número medido,
  nunca silenciada nem corrigida em código sem mudança de spec.
- GIVEN a EDA THEN não reimplementa nem repete provas do `0_pipeline_audit.ipynb`; referencia-o
  quando precisar de garantia de correção.

### REQ-109 — Check de balanço de covariáveis
WHEN os grupos tratado (viu a oferta) e controle (não viu) são comparados, the system SHALL
computar a diferença padronizada de médias (SMD) por covariável e sinalizar desbalanço acima
do limiar configurado.

Acceptance:
- GIVEN as covariáveis de cliente THEN cada uma tem SMD reportado.
- GIVEN o limiar configurado (default 0.1) THEN covariáveis acima dele são listadas como
  diagnóstico que qualifica a leitura causal (não altera o estimador — Premissa 5).

### REQ-110 — Configurabilidade
The system SHALL carregar todo parâmetro de comportamento (janelas, limiar SMD, regra de
prioridade de atribuição, cortes, seeds, caminhos) de um objeto de configuração validado, sem
valores hardcoded no corpo das funções.

Acceptance:
- GIVEN o código do pipeline THEN nenhum parâmetro de janela/limiar/caminho aparece literal
  fora do objeto de config.
- GIVEN uma config inválida (ex.: limiar negativo) THEN a carga falha na validação, antes de
  processar dados.

### REQ-111 — Segmentação de clientes por K-Means
WHEN a EDA segmenta clientes, the system SHALL ajustar um K-Means sobre features de **perfil e
volume de compra**, com a geometria correta para distância euclidiana, e reportar a resposta
observada (view, conversão, margem por envio) dentro de cada segmento.

Modelagem obrigatória, em `src/eda.cluster_matrix`:
- **Features:** `age`, `credit_card_limit`, `tenure_days`, `spend_total`, `txn_count`,
  `avg_ticket`. Nenhuma feature de resposta (`view_rate`, `conv_rate`, margem) entra no ajuste —
  clusterizar pela resposta e depois comparar resposta entre clusters é circular.
- **Sem imputação:** o segmento `identity_missing` não tem `age`/`credit_card_limit` (Premissa 3);
  fica fora do ajuste e volta na leitura como segmento nomeado. Imputar a mediana o empurraria
  para o centro do espaço e contaria a mesma ausência três vezes.
- **`log1p`** em `spend_total`, `txn_count`, `avg_ticket` antes de padronizar: a soma de quadrados
  euclidiana é dominada pela cauda sem isso.
- **z-score** após o log: sem escala comparável, `credit_card_limit` é o único eixo real.
- **`k`** escolhido por maior silhouette na faixa da config, com a varredura inteira (inércia e
  silhouette) reportada — nunca só o vencedor.

O rótulo de segmento é **descritivo**: usa a janela inteira do teste, logo não pode virar feature
do modelo da spec 02 (violaria G2).

Acceptance:
- GIVEN a matriz de ajuste THEN toda coluna tem média 0 e desvio 1, e nenhum nulo.
- GIVEN um cliente com `identity_missing=1` THEN ele não está na matriz de ajuste e aparece na
  leitura sob o segmento nomeado.
- GIVEN um nulo em cliente de perfil completo THEN a construção da matriz falha alto (G7 violada),
  em vez de imputar em silêncio.
- GIVEN a mesma `seed` THEN os rótulos são idênticos entre execuções.
- GIVEN a varredura de `k` THEN inércia e silhouette são reportadas para toda a faixa, e a
  silhouette é medida na mesma matriz padronizada e métrica do ajuste.
- GIVEN cada segmento × tipo de oferta THEN `envios`, `taxa_view`, `taxa_conversao`,
  `taxa_conversao_vistos` e `margem_por_envio` são reportados.

## Non-functional requirements

- **Integridade (crítico):** as garantias G1–G10 são cobertas por testes que falham a build se
  violadas. A suíte roda por CLI e é pré-requisito de qualquer entrega do dataset.
- **Reprodutibilidade:** todo o caminho bruto→processado roda por comando CLI único, ambiente
  gerenciado por UV, seeds fixas na config.
- **Visualização:** figuras em Plotly, padrão executivo — paleta sóbria e limitada, sem
  gridlines densas, rótulos diretos, título que afirma a conclusão, no máximo o necessário na
  tela. Legibilidade e baixa carga visual acima de ornamento.
- **Notebooks apenas para análise:** lógica de transformação vive em `src/`; notebooks importam
  de `src/` e servem só para exibir análises e figuras.

## Domain entities

- **Evento**: `account_id`, `event`, `time`, `offer_ref?`, `amount?`, `reward?` — a matéria bruta.
- **Oferta**: `offer_id`, `offer_type`, `duration`, `min_value`, `discount_value`, `channels`.
- **Cliente**: `account_id`, `age?`, `gender`, `credit_card_limit?`, `tenure_days`, `identity_missing`.
- **Linha processada**: o grão do contrato (schema-processed.md).

## Assumptions & open questions

- Premissas 1–4 e 7 de 00-clarify.md aplicam-se diretamente aqui.
- Sem clarificações pendentes em escopo obrigatório.

## Success criteria

- Dataset processado escrito, grão único, todas as garantias passando por teste.
- EDA permite a um revisor entender os dados e o enquadramento uplift em uma leitura.
- Diagnóstico de balanço disponível para a spec de modelagem.
