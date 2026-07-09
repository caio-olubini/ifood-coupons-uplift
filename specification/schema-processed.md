# schema-processed — contrato entre pipeline e modelagem

> Status: Ready
> Este é o **contrato**: a saída da spec 01 (pipeline) e a entrada da spec 02 (modelagem).
> Mudança aqui é mudança de interface — atualize ambas as specs, nunca só o código.

## Grão

Uma linha por **(cliente, oferta recebida)**. Cada evento `offer received` gera exatamente
uma linha. A chave `(account_id, offer_id, received_time)` é única — zero duplicatas.

Transações puras (sem oferta) e ofertas não-recebidas não geram linha; alimentam features
e atribuição, não o grão.

## Colunas

### Identificação e tratamento
| Coluna | Tipo | Descrição |
|---|---|---|
| `account_id` | string | cliente |
| `offer_id` | string | oferta recebida |
| `offer_type` | string | `bogo` \| `discount` \| `informational` |
| `received_time` | double | dia do recebimento (t desde início do teste) |
| `campaign_wave` | int | onda de campanha (0..5) derivada de `received_time` |
| `treatment` | int | tratamento efetivo: 1 se a oferta foi **vista**, 0 caso contrário |

`treatment` codifica exposição real, não recebimento. Uma oferta recebida e não vista não
expôs o cliente ao estímulo — é controle para fins de uplift.

### Label
| Coluna | Tipo | Descrição |
|---|---|---|
| `converted` | int | 1 se conversão influence-aware (Premissa 2): vista E transação **após o view** dentro da validade; senão 0 |
| `conversion_value` | double | soma das transações atribuídas (pós-view, na validade); 0 se não converteu |
| `reward_cost` | double | custo do desconto concedido (0 para informational e não-convertidos) |

### Features de cliente (do profile, tratando nulos)
| Coluna | Tipo | Descrição |
|---|---|---|
| `age` | int (nullable) | idade; **null** quando sentinela 118 (Premissa 3) |
| `gender` | string | `M` \| `F` \| `O` \| `unknown` |
| `credit_card_limit` | double (nullable) | limite; null preservado |
| `identity_missing` | int | 1 para o segmento sentinela (Premissa 3) |
| `tenure_days` | int | dias entre `registered_on` e t=0 |

### Features transacionais (pré-`received_time`, anti-leakage)
| Coluna | Tipo |
|---|---|
| `hist_spend_total` | double |
| `hist_txn_count` | int |
| `hist_avg_ticket` | double |
| `hist_spend_std` | double |
| `hist_recency_days` | double |
| `hist_frequency` | double |
| `hist_spend_trend` | double |

### Features de histórico de resposta a ofertas (pré-`received_time`)
| Coluna | Tipo | Descrição |
|---|---|---|
| `hist_offers_received` | int | por tipo também: `_bogo`, `_discount`, `_info` |
| `hist_offers_viewed` | int | |
| `hist_offers_completed` | int | |
| `hist_view_rate` | double | |
| `hist_conv_rate_bogo` | double | |
| `hist_conv_rate_discount` | double | |
| `hist_completed_unseen_flag` | int | já completou sem ver — assinatura de sure thing |
| `hist_time_view_to_conv` | double | tempo médio view→conversão |

### Features da oferta e contexto
| Coluna | Tipo |
|---|---|
| `discount_value` | double |
| `min_value` | double |
| `duration` | double |
| `n_channels` | int |
| `channel_web` / `_email` / `_mobile` / `_social` | int (0/1) |
| `discount_to_minvalue_ratio` | double |
| `n_concurrent_offers` | int |

## Garantias de integridade (verificáveis por teste)

Cada garantia é um invariante que, se violado, quebra o projeto em silêncio. São a espinha
da suíte de testes do pipeline.

- **G1 — Grão único.** Zero duplicatas em `(account_id, offer_id, received_time)`.
- **G2 — Sem leakage temporal.** Nenhuma feature `hist_*` incorpora evento com
  `time > received_time` da própria linha.
- **G3 — Label exige view.** `converted=1` ⇒ existe `offer viewed` com
  `view_time ≥ received_time` e `view_time ≤ received_time + duration`.
- **G4 — Conversão é pós-view e dentro da validade (influence-aware estrito).**
  `converted=1` ⇒ a transação atribuída ocorre em `[view_time, received_time + duration]`,
  isto é, **depois do view** e dentro da validade. Uma compra na janela mas
  anterior ao view não pode ter sido induzida pela visualização, logo não conta
  como conversão nem entra em `conversion_value`.
- **G5 — Informational sem completed.** Conversão de informational vem de transação em
  janela pós-view, nunca de evento `offer completed` (que não existe para esse tipo).
- **G6 — Custo coerente.** `reward_cost > 0` ⇒ `converted=1` e `offer_type ≠ informational`.
- **G7 — Sentinela tratada.** `age` nunca vale 118; `identity_missing=1` ⇔ os três campos
  de perfil ausentes.
- **G8 — Sem nulo em coluna não-nullable.** Apenas `age` e `credit_card_limit` admitem null.

## Encarnação executável

O contrato tem duas formas que devem concordar:
- **Spark `StructType`** — schema imposto na escrita do dataset processado.
- **Modelo Pydantic** — valida o schema e uma **amostra** do output (não linha a linha em
  massa). Falha alto se o pipeline entregar algo fora deste contrato.

Divergência entre as duas formas é um defeito de contrato.
