# Tasks: Simulador de alocação de cupons (MVP)

> Implementa: 03-simulator/spec.md
> Legenda: `[ ]` todo · `[~]` andamento · `[x]` feito · `[!]` bloqueado

**Ordem de implementação é incremental por fatia vertical, não por camada.** Cada
task (exceto T-301, config) termina com `simulator/index.html` abrindo no browser e
mostrando algo real e checável — nunca "export pronto mas UI ainda não existe" ou
vice-versa. Isso significa que o export de cada rodada exporta só o que aquela fatia
precisa; colunas/artefatos que faltam (τ, quadrante, `holdout.json`, ...) chegam nas
rodadas seguintes, cada uma ampliando o `matrix.json`/`metadata.json` anteriores em
vez de escrevê-los do zero. `simulator.export` é reexecutado a cada task; não há uma
task "escreve o export inteiro" isolada de UI.

**Restrições transversais (valem em toda task de UI, desde T-302):**
- **Identidade visual iFood desde o esqueleto** (REQ-312): a estrutura base
  (cabeçalho vermelho de marca, tipografia, layout, superfície neutra) é montada em
  T-302 e refinada ao longo do caminho — nunca "UI feia agora, bonita no fim". Cada
  fatia herda o design já estabelecido.
- **Linguagem para público não-técnico** (REQ-313): todo rótulo de controle, tooltip e
  texto de tela é escrito para leigos desde a primeira aparição. Nada de "τ",
  "uplift" cru, "Qini" na tela; termo técnico só com tradução ao lado.

**Modelo de interação (v2, a partir de T-305):** o usuário **escolhe uma estratégia**
(aleatório/conversão/uplift), **marca filtros** (tipo de oferta, quadrante) e clica em
**Gerar**. Gerar produz duas coisas: a **audiência** da estratégia escolhida (lista +
lucro projetado + download CSV) e o **card de analytics** comparando as três estratégias
sobre o holdout filtrado pelos mesmos filtros. Alvo/segmento **não** são estratégias —
são filtros. O export passa a gerar dois artefatos: `matrix.json` (serving, sem rótulo)
e `holdout.json` (rotulado, para o analytics). `validation.json` sai de escopo.

---

## T-301 — Config do simulador
- **Status:** [x]
- **Satisfies:** REQ-309
- **Depends on:** T-201 (config base da spec 02)
- **Files:** `src/config.py`, `tests/test_config_simulator.py`
- **Do:** Estender `PipelineConfig` com `simulator_output_dir`,
  `simulator_default_budget`, `simulator_score_gamma`, `simulator_temperature_default`,
  `simulator_temperature_max`. Validação na carga.
- **Accept:** valores ausentes caem em default documentado; tipo inválido falha antes
  do export. (Única task sem entregável visual — pré-requisito de todas as outras.)

## T-302 — Esqueleto: export mínimo + shell visual iFood + lista renderizada
- **Status:** [x]
- **Satisfies:** REQ-301 (item 1, parcial), REQ-305 (parcial), REQ-311 (parcial),
  REQ-312 (shell base)
- **Depends on:** T-301
- **Files:** `simulator/export.py`, `simulator/index.html`, `simulator/assets/`
  (CSS/fonte autohospedada, se houver), `simulator/data/` (gerado)
- **Do:** `simulator.export` monta `serve.build_scoring_frame` com todas as ofertas
  (bogo/discount/informational, sem score ainda) e escreve `matrix.json` (colunar:
  `account_id`, `offer_id`, `offer_type`) + `offers.json` (catálogo) +
  `metadata.json` (`decision_time`, `n_clients`, `seed`). `simulator/index.html` já
  nasce com o **shell visual iFood** (REQ-312): cabeçalho com vermelho de marca
  `#EA1D2C`, tipografia sem serifa (Inter/autohospedada), superfície neutra clara,
  layout com hierarquia. Faz `fetch` relativo dos três arquivos e renderiza uma
  tabela simples com as primeiras N linhas — sem knobs, sem alocação, só prova que
  o fetch/parse/render funciona ponta a ponta como estático, já dentro da identidade.
- **Accept:** abrir `simulator/index.html` via `python -m http.server` mostra a
  página com identidade iFood reconhecível e uma tabela com dados reais do holdout;
  nenhum erro de console; paths relativos (REQ-311); nenhuma fonte/asset de CDN
  externo em runtime.

## T-303 — Alocação básica: score aleatório + argmax + corte por budget
- **Status:** [x]
- **Satisfies:** REQ-302 (linha `aleatorio`), REQ-303 (checkboxes de oferta), REQ-304
  (passos 1, 2 e 4 — sem temperatura ainda)
- **Depends on:** T-302
- **Files:** `simulator/index.html`
- **Do:** Em JS: filtro de tipo de oferta (checkboxes bogo/discount/informational) →
  melhor oferta por cliente sob score **aleatório** seedado (argmax de um score
  pseudo-aleatório por `account_id`, desempate por `offer_id`) → corte pelos
  primeiros `budget` (campo numérico simples). Renderiza a lista final (conta de
  linhas + primeiras N) em vez da tabela crua de T-302.
- **Accept:** mudar o budget muda a contagem exibida; desmarcar um tipo de oferta
  redistribui clientes para a melhor oferta restante (REQ-303); resultado
  determinístico entre reloads com o mesmo seed.

## T-304 — Estratégias de conversão e uplift (modelos reais)
- **Status:** [x]
- **Satisfies:** REQ-301 (itens 2 e 3), REQ-302 (estratégias `conversao`/`uplift`)
- **Depends on:** T-303
- **Files:** `simulator/export.py`, `simulator/index.html`
- **Do:** Export passa a pontuar linhas bogo/discount com `BlendedUpliftModel`
  carregado (`uplift`, `p_convert`, `uncertainty`, `score_dynamic` γ=1,0) e a
  classificar quadrante (`quadrant.classify_quadrant`); `matrix.json` ganha essas
  colunas (informational continua com `uplift=null`). UI ganha o seletor de
  **estratégia** (aleatório/conversão/uplift) reutilizando o argmax+corte de T-303 com
  a coluna escolhida. Nota: estratégia é *só* o critério de ranqueamento — alvo/segmento
  entram como filtros em T-305, não como opção de estratégia.
- **Accept:** trocar a estratégia muda a lista resultante de forma visivelmente
  diferente (uplift prioriza τ alto, conversão prioriza p_convert alto); informational
  nunca entra no argmax sob `conversao`/`uplift` (aviso visível, REQ-303).

## T-305 — Filtros marcáveis + botão Gerar + temperatura
- **Status:** [x]
- **Satisfies:** REQ-304 (passo 3 + disparo por Gerar), REQ-306
- **Depends on:** T-304
- **Files:** `simulator/index.html`
- **Do:** Introduzir o **botão Gerar** como disparo explícito da alocação — a lista
  deixa de recalcular a cada tecla (refinamento sobre T-303: a lógica de alocação é a
  mesma, muda só o gatilho). Filtros de alvo como **checkboxes leigos** ("só quem o
  cupom convence" = persuadables; "excluir quem compraria sem cupom" = sem sleeping
  dogs), usando a coluna `quadrant` de T-304, aplicados antes do argmax; nenhum marcado
  = público inteiro, os dois marcados compõem. Slider de temperatura: `0` mantém a
  ordenação determinística; `>0` aplica Gumbel-max sobre o score min-max normalizado,
  PRNG seedado (seed em `metadata.json`).
- **Accept:** a lista só muda ao clicar em Gerar; com `temperature=0` a lista é idêntica
  à de T-304 (regressão zero); `temperature>0` produz ordens diferentes mas
  reprodutíveis com o mesmo seed; marcar "só persuadables" restringe a audiência a
  persuadable; público filtrado menor que o budget mostra "público esgotado: K < N".

## T-306 — Lucro projetado
- **Status:** [x]
- **Satisfies:** REQ-301 (item 4), REQ-307
- **Depends on:** T-305
- **Files:** `simulator/export.py`, `simulator/index.html`
- **Do:** Export calcula `lucro_medio_por_conversao_tratada` no holdout
  (`gaincurve._profit_per_treated_conversion`) e grava em `metadata.json`. UI exibe,
  para o top-N corrente: conversões incrementais esperadas (Σ τ), lucro incremental
  projetado (Σ τ × lucro médio do metadata) e orçamento de desconto projetado
  (Σ p_convert × discount_value, linha informativa separada), com o rótulo de
  projeção obrigatório.
- **Accept:** budget maior nunca reduz conversões esperadas; trocar a estratégia
  conversão↔uplift no mesmo budget muda os painéis de forma explicável; numa fixture
  pequena o lucro projetado bate com a soma manual (verificado por T-309).

## T-307 — Export do holdout rotulado + card de analytics comparativo
- **Status:** [x]
- **Satisfies:** REQ-301 (holdout), REQ-305 (`holdout.json`), REQ-308
- **Depends on:** T-306
- **Files:** `simulator/export.py` (holdout.json), `simulator/index.html`
- **Do:** Export gera `holdout.json` (colunar, rotulado: `account_id`, `offer_type`,
  `treatment`, `converted`, `net_profit` via `gaincurve.add_net_profit`, os três scores
  `score_random`/`p_convert`/`score_dynamic`, `quadrant`) e grava `analytics_budgets`
  (de `cfg.gain_curve_budgets`) no metadata. UI: ao clicar em **Gerar**, para cada uma
  das três estratégias, filtra o holdout pelos mesmos filtros marcados, ranqueia pela
  coluna de score da estratégia e calcula a curva de ganho incremental (espelhando
  `gaincurve.incremental_gain_curve` — contrafactual escalado estilo Qini) nos
  `analytics_budgets`; exibe a comparação no **card de analytics**. Também os 3 presets
  de estratégia (status quo / premiar / uplift) que setam os controles existentes.
- **Accept:** ao Gerar com filtros default, o card mostra as três estratégias nos
  budgets do metadata, uplift concentrando mais ganho; marcar "só persuadables" e Gerar
  de novo muda os números das três (medidos só sobre persuadables); a curva de cada
  estratégia bate com `gaincurve.incremental_gain_curve` numa fixture (verificado por
  T-309).

## T-308 — Seção "Sobre" + polimento final de design
- **Status:** [x]
- **Satisfies:** REQ-312 (polimento), REQ-313
- **Depends on:** T-307
- **Files:** `simulator/index.html`, `simulator/assets/`
- **Do:** Escrever a seção **"Sobre"** (aba ou painel expansível): o que é o
  simulador, o **fluxo Gerar** (escolher estratégia → marcar filtros → Gerar → lista +
  analytics + download), como cada estratégia funciona **na prática** (em linguagem
  para leigos, ver REQ-313), filtros como restrições (não modos), projeção vs.
  analytics, os quadrantes como "tipos de cliente", e a nota honesta de limites
  (avaliação offline, "quem viu" é escolha do cliente). Com as estratégias e o analytics
  já existentes (T-303…T-307), agora é possível descrever cada uma pelo que ela de fato
  faz na tela. Passar o design (REQ-312) por um polimento final: espaçamento,
  hierarquia, estados de foco/hover, contraste WCAG AA, responsividade em
  laptop/projetor, revisar toda a redação da UI para garantir zero jargão cru.
- **Accept:** um leigo entende cada estratégia, o fluxo Gerar e a diferença
  projeção/analytics lendo só a seção "Sobre"; nenhum termo técnico sem tradução na
  tela; a página está visualmente coesa e agradável, com identidade iFood consolidada.

## T-309 — Download da campanha escolhida (CSV)
- **Status:** [x]
- **Satisfies:** REQ-314
- **Depends on:** T-305 (a audiência já é congelada pelo Gerar)
- **Files:** `simulator/index.html`
- **Do:** Botão "Baixar campanha (CSV)" que exporta a audiência da estratégia escolhida
  (a lista gerada), gerado no browser via Blob, colunas de
  `serve.RECOMMENDATION_COLUMNS` (`rank`, `account_id`, `offer_id`, `offer_type`,
  `score`), na ordem de `rank`. Só disponível após Gerar.
- **Accept:** o CSV baixado tem exatamente as linhas exibidas, nas colunas e ordem de
  `RECOMMENDATION_COLUMNS`; antes de Gerar não há download; sem backend (Blob local).

## T-310 — Smoke test de paridade e export
- **Status:** [x]
- **Satisfies:** REQ-304, REQ-307, REQ-308, REQ-310
- **Depends on:** T-307
- **Files:** `tests/test_simulator_export.py`
- **Do:** Fixture pequena → export → (1) comparar top-N determinístico da audiência com
  `serve.recommend`; (2) **comparar a curva de ganho por estratégia com
  `gaincurve.incremental_gain_curve`** (o espelho JS do analytics não pode divergir do
  `src/`) — reimplementação do algoritmo JS em Python no teste, ou saída de referência;
  (3) verificar lucro projetado = soma manual; (4) verificar que linhas informational
  nunca entram no argmax de estratégia modelada.
- **Accept:** suíte passa; qualquer quebra de paridade (audiência ou ganho) ou de
  fórmula do lucro falha o teste. (Task de garantia, sem mudança visual na UI — a fatia
  vertical já foi entregue em T-303…T-307; aqui só se trava o que já está de pé.)

## T-311 — README e entrypoint
- **Status:** [x]
- **Satisfies:** REQ-310
- **Depends on:** T-307
- **Files:** `simulator/README.md`
- **Do:** Documentar os 4 passos: `pipeline` → `train` → `python -m simulator.export`
  → `python -m http.server` em `simulator/` (teste local).
- **Accept:** um revisor sem contexto prévio consegue rodar o simulador seguindo só o
  README.

## T-312 — Publicação no GitHub Pages
- **Status:** [x]
- **Satisfies:** REQ-311
- **Depends on:** T-308, T-311
- **Files:** `simulator/index.html`, `simulator/data/`, `.github/` (config de Pages,
  se necessário), `simulator/README.md`
- **Do:** Confirmar que todo `fetch`/link em `simulator/index.html` já é path
  relativo (deveria ser desde T-302); confirmar que nenhum arquivo de
  `simulator/data/` (incluindo `holdout.json`) passa de 50 MB; commitar os artefatos
  gerados; habilitar GitHub Pages apontando para a pasta (ou branch) certa; documentar
  a URL final no README.
- **Accept:** a página publicada carrega os JSONs, os controles funcionam, Gerar produz
  audiência + analytics, e não há erro de CORS/path absoluto no console do browser.

## T-313 — Adicional (se sobrar tempo): filtro por segmento K-Means
- **Status:** [ ]
- **Satisfies:** seção "Adicional" de spec.md
- **Depends on:** T-307
- **Files:** `simulator/export.py`, `simulator/index.html`
- **Do:** Export anexa a coluna de segmento K-Means por cliente (`eda.cluster_matrix →
  fit_clusters → assign_segments`) **na `matrix.json` e no `holdout.json`** (o filtro
  age nos dois, como o quadrante); a UI ganha **mais um campo marcável** de filtro de
  alvo (select de segmento), compondo com os filtros de quadrante para a audiência e
  para o analytics.
- **Accept:** marcar um segmento restringe a audiência e o analytics ao segmento
  escolhido antes do argmax; não bloqueia o DoD do MVP se não implementado.
