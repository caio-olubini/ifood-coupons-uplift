# Catálogo de testes — pipeline de dados

> Cobertura da suíte de integridade do pipeline (`tests/`). Cada teste guarda um
> invariante que, se quebrado, corromperia o dataset **em silêncio** — a
> modelagem produziria números plausíveis e errados. Rode com `uv run pytest -q`.

## Filosofia

- **Fixtures sintéticas, não amostras reais.** Cada teste monta o dataset mínimo
  que exercita exatamente uma falha, determinístico e legível. O dataset real é
  validado à parte, no notebook de pipeline.
- **Um teste, uma mina.** Se um teste é removido, uma armadilha específica volta
  a passar despercebida. Nenhum é decorativo.
- **As garantias G1–G8** (ver `schema-processed.md`) são a espinha da suíte;
  cada uma tem teste dedicado.

## Mapa garantia → teste

| Garantia | O que protege | Coberta por |
|---|---|---|
| **G1** grão único | zero duplicatas em `(account_id, offer_id, received_time)` | `test_g1_unique_grain_no_duplicates`, `test_g1_same_offer_two_waves_both_survive` |
| **G2** sem leakage temporal | nenhuma feature `hist_*` usa evento com `time ≥ received_time` | `test_post_receipt_transaction_does_not_leak_into_hist_features`, `test_post_receipt_offer_events_do_not_leak` |
| **G3** label exige view | `converted=1` ⇒ view dentro da janela | `test_completed_without_prior_view_is_not_converted` |
| **G4** conversão pós-view e na validade | `converted=1` ⇒ transação em `[view_time, valid_until]` (após o view) | `test_transaction_after_validity_does_not_convert`, `test_transaction_before_view_does_not_convert`, `test_transaction_after_view_converts` |
| **G5** informational sem `completed` | conversão de informational vem da janela pós-view | `test_informational_converts_from_post_view_window_not_completed_event`, `test_informational_conversion_has_zero_cost` |
| **G6** custo coerente | `reward_cost > 0` ⇒ convertido e não-informational | `test_g6_invariant_holds_across_rows` (+ 3 casos em `test_cost.py`) |
| **G7** sentinela tratada | `identity_missing=1` ⇔ os três campos ausentes; `age` nunca 118 | `test_g7_identity_missing_iff_three_fields_absent`, `test_sentinel_gets_identity_missing_and_null_age` |
| **G8** sem nulo indevido | só `age` e `credit_card_limit` admitem null no contrato | `test_g8_no_nulls_in_non_nullable_columns` |

## Armadilhas conhecidas do dataset e sua cobertura

Estas são as minas específicas deste dataset — cada uma pegaria um pipeline
ingênuo. Todas têm teste.

| Armadilha | Por que engana | Teste |
|---|---|---|
| **Dois nomes para a oferta** (`offer id` vs `offer_id`) | ler só um perde 100% de received/viewed ou de completed | `test_received_and_viewed_read_offer_id_with_space`, `test_completed_reads_offer_id_with_underscore` |
| **Idade 118 = identidade ausente** | tratar como idade real polui estatísticas; imputar destrói o segmento | `test_g7_identity_missing_iff_three_fields_absent` |
| **Completou sem ver ("sure thing")** | contar como conversão infla o efeito; a compra aconteceria de todo modo | `test_completed_without_prior_view_is_not_converted` |
| **Informational sem `offer completed`** | esperar o evento zera a conversão desse tipo; ele não existe para informational | `test_informational_converts_from_post_view_window_not_completed_event` |
| **View da onda errada** (mesma oferta em ondas) | um view pode ser atribuído a um recebimento anterior cuja janela não o contém, roubando a conversão da onda certa | `test_view_attributed_to_containing_receipt_window_across_waves` |
| **Comprou antes de ver** | contar como conversão uma compra anterior ao view atribui à oferta um efeito que ela não causou (11,4% das conversões, na leitura frouxa) | `test_transaction_before_view_does_not_convert` |
| **Múltiplas transações na janela** | tratar cada transação como uma linha duplica o grão (bug real já corrigido) | `test_g1_unique_grain_no_duplicates` |
| **Ofertas sobrepostas** (Premissa 1) | atribuir a mesma transação a duas ofertas conta em dobro | `test_overlapping_offers_apply_configured_priority` |
| **Leakage temporal sutil** | um evento pós-recebimento numa feature de janela vaza o futuro | `test_post_receipt_transaction_does_not_leak_into_hist_features` |
| **Cliente sem histórico** | join que remove a linha sem eventos, ou null onde deveria ser 0 | `test_no_history_yields_zeroed_counts` |
| **Config inválida** | limiar/janela absurdos passam silenciosamente para o processamento | `test_negative_smd_threshold_fails_at_load`, `test_zero_campaign_wave_days_fails_at_load` |

## Testes por arquivo

### `tests/test_config.py` — carga da config (REQ-110)
- **`test_default_config_loads`** — a config padrão carrega e traz os defaults (limiar 0.1, sentinela 118).
- **`test_negative_smd_threshold_fails_at_load`** — limiar SMD negativo levanta erro **na carga**, antes de tocar em dados.
- **`test_zero_campaign_wave_days_fails_at_load`** — janela de onda ≤ 0 falha na carga.

### `tests/test_parsing.py` — leitura e coalescência de `value` (REQ-101)
- **`test_received_and_viewed_read_offer_id_with_space`** — received/viewed extraem a oferta de `offer id` (com espaço), não-nula.
- **`test_completed_reads_offer_id_with_underscore`** — completed extrai de `offer_id` (com underscore) e traz `reward`.
- **`test_transaction_has_amount_and_no_offer_ref`** — transaction traz `amount` e nenhuma referência de oferta.

### `tests/test_profile.py` — normalização de perfil (REQ-102)
- **`test_sentinel_gets_identity_missing_and_null_age`** — o sentinela recebe `identity_missing=1` e `age=null`; o normal não.
- **`test_missing_gender_becomes_unknown`** — `gender` ausente vira `unknown`.
- **`test_tenure_days_is_nonnegative_integer`** — `tenure_days` é o intervalo correto até `test_start_date`.

### `tests/test_attribution.py` — atribuição temporal (REQ-103)
- **`test_transaction_outside_validity_window_not_assigned`** — transação fora da janela não é atribuída; a linha do grão sobrevive com contagem 0.
- **`test_transaction_inside_validity_window_assigned`** — transação na janela é atribuída (contagem, soma, primeiro tempo).
- **`test_overlapping_offers_apply_configured_priority`** — em sobreposição, a `AttributionPriority` da config decide qual oferta fica com a transação (testa `earliest` e `latest`).

### `tests/test_label.py` — label influence-aware (REQ-104)
- **`test_completed_without_prior_view_is_not_converted`** — sem view precedente, não converte (G3).
- **`test_transaction_after_validity_does_not_convert`** — transação após o fim da validade não converte (G4).
- **`test_informational_converts_from_post_view_window_not_completed_event`** — informational converte pela janela pós-view, sem `offer completed` (G5).
- **`test_transaction_before_view_does_not_convert`** — compra anterior ao view (mesmo na validade) não converte: não pode ter sido induzida pela visualização (G4 estrito).
- **`test_transaction_after_view_converts`** — mesma situação, mas a compra vem depois do view ⇒ converte.
- **`test_view_attributed_to_containing_receipt_window_across_waves`** — na mesma oferta recebida em duas ondas, o view pertence à onda cuja janela o contém; a onda errada não rouba a conversão.
- **`test_viewed_and_in_window_transaction_converts`** — caso positivo canônico: vista + transação na janela ⇒ converte, com valor.

### `tests/test_leakage.py` — features anti-leakage (REQ-105, G2)
- **`test_post_receipt_transaction_does_not_leak_into_hist_features`** — transação posterior ao recebimento (valor gritante) **não** entra em `hist_spend_total`/count/ticket.
- **`test_no_history_yields_zeroed_counts`** — cliente sem histórico: contadores zerados, linha preservada (não vira null nem some).
- **`test_post_receipt_offer_events_do_not_leak`** — um recebimento posterior não infla `hist_offers_received` da linha anterior.
- **`test_offer_context_features_from_catalog`** — features de contexto (`n_channels`, `channel_*`, `discount_to_minvalue_ratio`) saem corretas do catálogo.

### `tests/test_cost.py` — custo do desconto (REQ-106, G6)
- **`test_converted_bogo_has_positive_cost`** — conversão de bogo recebe `reward_cost = discount_value`.
- **`test_informational_conversion_has_zero_cost`** — informational converte mas com custo 0.
- **`test_not_converted_has_zero_cost`** — não-convertido tem custo 0.
- **`test_g6_invariant_holds_across_rows`** — globalmente: `reward_cost > 0` ⇒ convertido e não-informational.

### `tests/test_integrity.py` — invariantes end-to-end (G1, G7, G8)
- **`test_g1_unique_grain_no_duplicates`** — após o pipeline completo, zero duplicatas na chave do grão (inclui dois recebimentos da mesma oferta).
- **`test_g1_same_offer_two_waves_both_survive`** — duas ondas legítimas da mesma oferta geram duas linhas distintas, ambas presentes.
- **`test_g7_identity_missing_iff_three_fields_absent`** — o **⇔** da sentinela: a flag implica os três campos ausentes e vice-versa; `age` nunca vale 118 após normalização.
- **`test_g8_no_nulls_in_non_nullable_columns`** — nenhuma coluna do contrato fora de `age`/`credit_card_limit` contém null (colunas intermediárias e features com null semântico são excluídas explicitamente).

## Lacunas conhecidas (a fechar com as próximas tasks)

- **G1/G8 pelo contrato imposto (T-108).** Hoje G1 e G8 são testados sobre a
  saída intermediária do pipeline. Quando `src/contract.py` impuser o
  `StructType` e a validação Pydantic de amostra, deve haver um teste que
  valide o dataset **escrito em `data/processed/`** contra o contrato — o ponto
  em que uma divergência de schema seria fatal.
- **Fixture central compartilhada.** Cada arquivo define `_setup`/`_offer`
  locais. Ao crescer, vale extrair um builder de eventos único em `conftest.py`.
