# schema-raw — glossário dos dados brutos

> Descreve os três arquivos de entrada em `data/raw/`, antes de qualquer
> transformação. Ver `schema-processed.md` para o contrato de saída do pipeline.

## `offers.json` — catálogo de ofertas

10 registros. Cada linha é **uma oferta possível de ser enviada**, não um envio.

| Campo | Tipo | Descrição |
|---|---|---|
| `id` | string | identificador único da oferta. |
| `offer_type` | string | `bogo` (buy-one-get-one) \| `discount` \| `informational`. Informational não tem recompensa por completar — é só uma mensagem/anúncio. |
| `discount_value` | double | valor do desconto ou da recompensa concedida ao completar. `0` para informational. |
| `min_value` | double | gasto mínimo do cliente para a oferta valer (ativar o desconto/recompensa). `0` para informational (não há gatilho de valor). |
| `duration` | double | quantos dias a oferta fica válida a partir do recebimento. |
| `channels` | array\<string\> | canais de veiculação: `web`, `email`, `mobile`, `social`. |

Distribuição real: 4 `bogo`, 4 `discount`, 2 `informational`.

## `profile.json` — cadastro de clientes

17.000 registros. Um por cliente.

| Campo | Tipo | Descrição |
|---|---|---|
| `id` | string | identificador único do cliente (`account_id` nos outros arquivos). |
| `age` | int | idade declarada. **`118` é sentinela de identidade ausente**, não idade real (ver Premissa 3 em `00-clarify.md`) — 2.175 clientes (12,8%), sempre com `gender=null` e `credit_card_limit=null` simultaneamente. |
| `gender` | string \| null | `M` \| `F` \| `O` \| `null` (ausente, coincide com a sentinela de idade). |
| `credit_card_limit` | double \| null | limite de cartão; `null` coincide com a sentinela. |
| `registered_on` | string `YYYYMMDD` | data de cadastro do cliente no app. |

## `transactions.json` — log de eventos

306.534 registros. É um **log de eventos heterogêneo**: cada linha é um evento de um dos quatro tipos abaixo, todos misturados no mesmo arquivo e diferenciados pelo campo `event`. Apesar do nome do arquivo, a maior parte das linhas não são compras — são eventos do ciclo de vida de uma oferta.

| Campo | Tipo | Descrição |
|---|---|---|
| `account_id` | string | cliente a quem o evento pertence. |
| `event` | string | um de `offer received`, `offer viewed`, `offer completed`, `transaction`. |
| `time_since_test_start` | double | tempo do evento, em dias desde o início do experimento (`t=0`). Varia de 0 a 29,75 no dataset. |
| `value` | object | payload específico do tipo de evento — ver subcampos abaixo. **Nunca todos preenchidos ao mesmo tempo**; cada tipo de evento usa um subconjunto. |

### Subcampos de `value` (a "mina" do dataset)

| Subcampo | Preenchido em | Descrição |
|---|---|---|
| `offer id` (com espaço) | `offer received`, `offer viewed` | referência à oferta do catálogo. |
| `offer_id` (com underscore) | `offer completed` | **mesmo significado** que `offer id`, mas com nome de campo diferente — os dois nomes precisam ser coalescidos numa única referência ao ler o dado (ver `schema-processed.md`, REQ-101). |
| `amount` | `transaction` | valor gasto na compra. |
| `reward` | `offer completed` | valor da recompensa recebida ao completar (equivale ao `discount_value` da oferta, para bogo/discount). |

### Os quatro tipos de evento

- **`offer received`** — a oferta foi enviada ao cliente (não implica que ele viu). Contém `offer id`.
  76.277 eventos.
- **`offer viewed`** — o cliente abriu/visualizou a oferta recebida. Contém `offer id`. É o que define `treatment` no dataset processado (tratamento efetivo = exposição real, não recebimento). 57.725 eventos.
- **`offer completed`** — o cliente atingiu o gatilho de recompensa da oferta (gastou `min_value`). Contém `offer_id` e `reward`. **Só existe para `bogo`/`discount`** — informational nunca gera este evento, porque não tem recompensa a completar. 33.579 eventos.
- **`transaction`** — uma compra do cliente, com ou sem relação a alguma oferta. Contém `amount`, nunca referência de oferta. É o evento bruto a partir do qual o pipeline infere se uma compra deve ser atribuída a uma oferta ativa (atribuição temporal, REQ-103). 138.953 eventos.

### Por que `offer completed` não basta como label de conversão

Um cliente pode "completar" uma oferta (gastar o suficiente) sem nunca ter visto
o anúncio — nesse caso a compra teria acontecido de qualquer forma. Por isso o
pipeline usa um label **influence-aware**: só conta como conversão quando há
`offer viewed` antes da `transaction` dentro da janela de validade (Premissa 2).
Isso também é o motivo de `informational` conseguir "converter" sem nunca ter
um `offer completed` — a conversão vem da janela pós-view, não deste evento.
