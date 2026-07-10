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
- **As garantias G1–G10** (ver `schema-processed.md`) são a espinha da suíte;
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
| **G9** exposição exclusiva | uma view física é `view_time` de no máximo um recebimento | `test_single_view_serves_only_one_of_two_overlapping_receipts` |
| **G10** conversão atinge o mínimo | `converted=1` ⇒ `conversion_value ≥ min_value` | `test_g10_conversion_implies_min_value_reached`, `test_transaction_below_min_value_does_not_convert`, `test_ineligible_offer_does_not_steal_transaction_from_an_eligible_one` |

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
| **Compra abaixo do `min_value`** | contá-la como conversão debita um desconto que nunca seria concedido (25,9% das conversões pagas, antes de G10) | `test_transaction_below_min_value_does_not_convert`, `test_g10_conversion_implies_min_value_reached` |
| **Oferta inelegível vence a prioridade** | filtrar o `min_value` depois do desempate deixa a oferta cara roubar e descartar a transação que a barata converteria | `test_ineligible_offer_does_not_steal_transaction_from_an_eligible_one` |
| **Leakage temporal sutil** | um evento pós-recebimento numa feature de janela vaza o futuro | `test_post_receipt_transaction_does_not_leak_into_hist_features` |
| **Cliente sem histórico** | join que remove a linha sem eventos, ou null onde deveria ser 0 | `test_no_history_yields_zeroed_counts` |
| **Config inválida** | limiar/nº de ondas absurdos passam silenciosamente para o processamento | `test_negative_smd_threshold_fails_at_load`, `test_zero_campaign_waves_fails_at_load` |

## Testes por arquivo

### `tests/test_config.py` — carga da config (REQ-110)
- **`test_default_config_loads`** — a config padrão carrega e traz os defaults (limiar 0.1, sentinela 118).
- **`test_negative_smd_threshold_fails_at_load`** — limiar SMD negativo levanta erro **na carga**, antes de tocar em dados.
- **`test_zero_campaign_waves_fails_at_load`** — número de ondas ≤ 0 falha na carga.

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
- **`test_single_view_serves_only_one_of_two_overlapping_receipts`** — uma view física marca no máximo um recebimento (G9).
- **`test_unseen_offer_does_not_steal_transaction_from_a_seen_one`** — oferta não-vista nunca disputa a transação de uma vista.
- **`test_identical_received_time_resolves_deterministically_by_offer_id`** — `received_time` empatado desempata por `offer_id`, estável entre execuções.
- **`test_ineligible_offer_does_not_steal_transaction_from_an_eligible_one`** — o `min_value` é filtrado **antes** do desempate: a oferta inelegível não vence a prioridade para depois descartar a transação que a elegível converteria (G10).

### `tests/test_label.py` — label influence-aware (REQ-104)
- **`test_completed_without_prior_view_is_not_converted`** — sem view precedente, não converte (G3).
- **`test_transaction_after_validity_does_not_convert`** — transação após o fim da validade não converte (G4).
- **`test_informational_converts_from_post_view_window_not_completed_event`** — informational converte pela janela pós-view, sem `offer completed` (G5).
- **`test_transaction_before_view_does_not_convert`** — compra anterior ao view (mesmo na validade) não converte: não pode ter sido induzida pela visualização (G4 estrito).
- **`test_transaction_after_view_converts`** — mesma situação, mas a compra vem depois do view ⇒ converte.
- **`test_view_attributed_to_containing_receipt_window_across_waves`** — na mesma oferta recebida em duas ondas, o view pertence à onda cuja janela o contém; a onda errada não rouba a conversão.
- **`test_transaction_below_min_value_does_not_convert`** — compra pós-view na validade, abaixo do gasto mínimo, não converte nem entra em `conversion_value`; gastar exatamente o mínimo converteria (fronteira fechada) (G10).
- **`test_small_transactions_do_not_sum_past_min_value`** — o limiar é por transação, não sobre o gasto acumulado: duas compras de R$ 6 numa oferta de mínimo 10 não convertem.
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
- **`test_g6_invariant_holds_across_rows`** — globalmente: `reward_cost > 0` ⇒ convertido e não-informational. G10 (compra abaixo do mínimo não converte) é coberto a montante em `test_label.py` e ponta a ponta em `test_g10_conversion_implies_min_value_reached`.

### `tests/test_integrity.py` — invariantes end-to-end (G1, G7, G8, G10)
- **`test_g1_unique_grain_no_duplicates`** — após o pipeline completo, zero duplicatas na chave do grão (inclui dois recebimentos da mesma oferta).
- **`test_g1_same_offer_two_waves_both_survive`** — duas ondas legítimas da mesma oferta geram duas linhas distintas, ambas presentes.
- **`test_g10_conversion_implies_min_value_reached`** — no dataset inteiro, toda linha convertida tem `conversion_value ≥ min_value`, e a compra abaixo do mínimo não converte nem custa.
- **`test_g7_identity_missing_iff_three_fields_absent`** — o **⇔** da sentinela: a flag implica os três campos ausentes e vice-versa; `age` nunca vale 118 após normalização.
- **`test_g8_no_nulls_in_non_nullable_columns`** — nenhuma coluna do contrato fora de `age`/`credit_card_limit` contém null (colunas intermediárias e features com null semântico são excluídas explicitamente).

### `tests/test_contract.py` — contrato e escrita (REQ-107, T-108)
- **`test_struct_type_and_pydantic_share_the_same_columns`** — as duas formas do contrato saem da mesma lista canônica; divergir seria defeito de contrato.
- **`test_assembled_dataset_matches_contract_schema_exactly`** — nomes, ordem e tipos idênticos ao `StructType`.
- **`test_intermediate_columns_are_dropped_from_the_contract`** — `view_time`/`valid_until`/`assigned_*` não vazam para o dataset final.
- **`test_treatment_is_one_iff_offer_was_viewed`** — `treatment` codifica exposição, não recebimento.
- **`test_campaign_wave_is_the_zero_based_rank_of_distinct_received_times`** — a onda é o rank do disparo, não um bucket de largura fixa.
- **`test_g8_rejects_a_null_injected_into_a_non_nullable_column`** — G8 falha alto ao ver null onde o contrato proíbe.
- **`test_nullable_history_features_do_not_trip_g8`** — `hist_recency_days`/`hist_time_view_to_conv` são null semântico ("sem histórico"), não violação.
- **`test_validate_sample_raises_on_a_contract_violating_row`** — a validação Pydantic de amostra pega tipo fora do contrato.
- **`test_run_writes_parquet_that_reconforms_to_the_contract`** — o parquet escrito em `data/processed/` relido bate com o contrato.

### `tests/test_eda.py` — visões, balanço e segmentação (REQ-108, REQ-109, REQ-111, T-109/110/111/112)
- **`test_smd_matches_the_cohen_formula_on_known_values`** — o SMD bate com a fórmula de Cohen em valores conhecidos, à mão.
- **`test_zero_variance_covariate_yields_smd_zero_not_nan`** — variância nula com médias iguais dá SMD 0.0, não NaN (que sumiria da tabela de diagnóstico em silêncio).
- **`test_gender_enters_the_balance_as_one_indicator_per_level`** — bug real pego pelo teste: variância nula **com médias distintas** (separação perfeita) tem de dar |SMD| infinito, não 0.0 — a versão inicial devolvia 0.0 e escondia o pior desbalanço possível.
- **`test_assignment_balance_reports_worst_pair_over_received_offers`** — a verificação da Premissa 4 (envio aleatório) usa o `offer_id` recebido como grupo, não viu/não-viu (que é pós-tratamento); reporta o pior par entre ofertas.
- **`test_null_age_is_ignored_in_the_mean_not_treated_as_zero`** — nulo de perfil não vira 0 na média (não polui o SMD do segmento sentinela).
- **`test_completed_unseen_rate_per_offer_type`** / **`test_view_after_completion_does_not_count_as_seen`** — a taxa "completou sem ver" usa `view_time ≤ completed_time`, não presença de qualquer view.
- **`test_informational_appears_with_zero_completed_and_null_rate`** — informational aparece no relatório com taxa **nula** (não 0.0, que mentiria "sempre viu antes") — espelha G5.
- **`test_identity_null_overlap_counts_the_intersection`** — a interseção dos três campos ausentes é contada corretamente (Premissa 3).
- **`test_numeric_histogram_excludes_nulls_and_conserves_the_count`** / **`test_numeric_histogram_handles_a_constant_column`** — nulos não viram bucket; coluna constante (min==max) não estoura divisão por zero.
- **`test_campaign_waves_reports_view_rate_per_wave`** — a taxa de view por onda é `vistos/recebimentos`, não uma contagem crua.
- **`test_unattributable_share_counts_transactions_outside_every_window`** — a fração de compra espontânea (Ato 3) conta transações fora de QUALQUER janela de recebimento, mais ampla que o filtro do label (não depende de view).
- **`test_positivity_counts_clients_who_never_received_a_type`** — a cobertura por tipo de oferta (positividade/Premissa 8) conta clientes que nunca receberam aquele tipo, não os que não converteram.
- **`test_conversion_by_segment_splits_by_tenure_quartile`** — a heterogeneidade por segmento separa corretamente por quartil de `tenure_days`, com contraste verificável nos extremos.

Perfil univariado, outliers e correlação (REQ-108):

- **`test_numeric_profile_separates_null_from_zero`** — `frac_nulos` divide pelo total de linhas, `frac_zeros` pelas linhas com valor; trocar os denominadores é o jeito mais fácil de uma tabela descritiva mentir.
- **`test_numeric_profile_flags_the_tukey_fence_outlier`** — a cerca de Tukey marca a cauda e nada mais; `min`/`max` continuam sendo os valores reais.
- **`test_numeric_profile_survives_an_all_null_column`** — coluna inteiramente nula devolve quantis NaN e zero outliers, sem estourar.
- **`test_correlation_uses_pairwise_deletion_not_zero_fill`** — preencher nulo com 0 para caber num `VectorAssembler` inventaria correlação onde há só ausência; a exclusão par a par preserva a relação real.
- **`test_redundant_pairs_reports_each_pair_once_above_threshold`** — o triângulo superior, não a matriz inteira (senão todo par aparece duas vezes).
- **`test_sanity_checks_count_the_impossible_rows`** — as combinações que violariam G3/G6 viram contagem, não `assert` (a prova formal é do audit).
- **`test_response_funnel_keeps_the_two_denominators_apart`** / **`test_response_funnel_rate_over_viewed_is_null_when_nobody_viewed`** — taxa sobre recebidos ≠ taxa sobre vistos; denominador zero dá `nan`, jamais 0.0 (que leria "foi exposto e não converteu").

Segmentação K-Means (REQ-111):

- **`test_client_features_gives_zero_ticket_to_a_client_who_never_bought`** — "não comprou" é o valor 0, não ausência; e não há divisão por zero no ticket médio.
- **`test_cluster_matrix_leaves_the_sentinel_segment_out_of_the_fit`** — o segmento `identity_missing` não é imputado nem descartado: sai do ajuste e volta nomeado.
- **`test_cluster_matrix_standardizes_every_column`** — média 0, desvio 1: sem isso, `credit_card_limit` (milhares) seria o único eixo real da distância euclidiana.
- **`test_cluster_matrix_log_transforms_the_heavy_tails`** — `log1p` antes do z-score nas features de gasto; um cliente 100× mais gastador não pode ficar 100× mais longe.
- **`test_cluster_matrix_refuses_a_null_in_a_complete_profile_client`** — G7 violada falha alto aqui, não páginas adiante dentro do sklearn.
- **`test_cluster_matrix_refuses_a_constant_feature`** — desvio zero não pode virar divisão por zero silenciosa.
- **`test_fit_clusters_separates_two_obvious_blobs_and_is_deterministic`** — dois grupos afastados são separados, e a mesma seed devolve o mesmo particionamento.
- **`test_cluster_scan_covers_the_configured_range_and_picks_the_best_silhouette`** — inércia decresce com `k` (não é critério sozinha), silhouette fica em [-1,1] e `choose_k` devolve o argmax.
- **`test_assign_segments_names_the_sentinel_instead_of_leaving_it_null`** — o cliente sem perfil ganha nome de segmento, não `NaN` de cluster.
- **`test_segment_response_keeps_denominators_apart`** — conversão sobre envios ≠ sobre vistos; margem por envio é `(receita − custo)/envios`.
- **`test_window_spend_ignores_the_view_unlike_the_label`** — a régua de gasto na janela conta transação sem view (o label, por G4, não conta) — é essa a diferença que ela existe para medir.
- **`test_naive_spend_lift_is_the_raw_difference_viewed_minus_unviewed`** — diferença bruta, confundida por seleção; nunca um efeito causal.

## Lacunas conhecidas (a fechar com as próximas tasks)

- **Fixture central compartilhada.** Cada arquivo define `_setup`/`_offer`
  locais. Ao crescer, vale extrair um builder de eventos único em `conftest.py`.
