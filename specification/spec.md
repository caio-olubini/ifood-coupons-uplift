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
vista pelo mesmo cliente, the system SHALL atribuir essa transação àquela oferta, assumindo no
máximo uma oferta ativa por cliente no intervalo (Premissa 1).

Acceptance:
- GIVEN uma transação fora de qualquer janela de validade THEN não é atribuída a nenhuma oferta.
- GIVEN duas ofertas ativas no mesmo intervalo para um cliente THEN o pipeline aplica a regra de
  prioridade configurada e registra a ocorrência em log de premissa.

### REQ-104 — Label influence-aware
IF uma oferta foi vista e uma transação atribuída ocorre dentro da validade, THEN the system
SHALL marcar `converted=1`; caso contrário `converted=0`.

Acceptance:
- GIVEN uma oferta completada sem view precedente THEN `converted=0`.
- GIVEN uma oferta com transação após `received_time + duration` THEN `converted=0`.
- GIVEN informational vista com transação na janela pós-view THEN `converted=1` sem depender
  de evento `offer completed`.

### REQ-105 — Features sem leakage temporal
WHILE computa qualquer feature `hist_*` para uma linha, the system SHALL usar apenas eventos
com `time < received_time` daquela linha.

Acceptance:
- GIVEN qualquer linha do dataset THEN nenhuma feature incorpora evento em `time ≥ received_time`
  (garantia G2, verificada por teste dedicado).

### REQ-106 — Custo do desconto
WHEN uma conversão de bogo ou discount é registrada, the system SHALL preencher `reward_cost`
com o custo do desconto; para informational e não-convertidos, `reward_cost=0`.

Acceptance:
- GIVEN `reward_cost > 0` THEN `converted=1` e `offer_type ≠ informational` (garantia G6).

### REQ-107 — Escrita conforme contrato
WHEN o dataset processado é escrito, the system SHALL impor o `StructType` do contrato e passar
por validação Pydantic de schema e amostra.

Acceptance:
- GIVEN o dataset final THEN todas as garantias G1–G8 do contrato passam.
- GIVEN o dataset final THEN a validação Pydantic de amostra não levanta erro.

### REQ-108 — EDA legível
WHEN a EDA é executada, the system SHALL produzir as visões que permitem avaliar os dados:
distribuição dos quatro eventos no tempo, as seis ondas de campanha, taxa de "completou sem ver"
por tipo de oferta, sobreposição dos nulos, e distribuições de features-chave.

Acceptance:
- GIVEN a EDA THEN cada visão tem uma figura polida (ver requisito não-funcional de visualização).
- GIVEN a EDA THEN a taxa de completou-sem-ver é reportada por tipo de oferta.

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

## Non-functional requirements

- **Integridade (crítico):** as garantias G1–G8 são cobertas por testes que falham a build se
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
