# Spec: Simulador de alocação de cupons (MVP)

> **Status:** Draft
> **Consome:** modelos em `models_dir`, `config.yaml`, `data/raw/`, funções de
> `src/serve.py`, `src/models.py`, `src/gaincurve.py`, `src/quadrant.py`
> **Vive em:** pasta própria `simulator/` (fora de `src/`/`tools/`), referenciando
> livremente o resto do projeto — não é um módulo do pipeline.
> **Deploy alvo:** GitHub Pages — 100% estático, sem backend, sem build step; os
> artefatos do export são commitados e servidos como arquivos.
> **Princípio:** nenhuma fórmula nova — o simulador orquestra engenharia existente; o
> que roda no browser espelha `serve.recommend`, não o substitui.

## Purpose

Interface HTML estática que responde: **dado um budget de N cupons, a quem enviar, com
qual oferta e qual lógica** — e **quanto isso deve render** (lucro projetado), com um
**relatório que compara todas as estratégias** lado a lado sob as mesmas restrições.

O fluxo é uma ação explícita de produto:

```
escolher estratégia + marcar filtros → [Gerar]
   ├─ audiência: recommend(estratégia escolhida, budget, filtros)  → lista + lucro projetado + download CSV
   └─ analytics: para CADA estratégia, curva de ganho no holdout FILTRADO pelos mesmos filtros → comparação
```

A **estratégia** (aleatório / conversão / uplift) escolhe *como ranquear*; os **filtros**
(tipo de campanha, quadrante de cliente, e — adicional — segmento) restringem *o público*
e valem para a audiência **e** para todas as estratégias do relatório. **Alvo não é uma
estratégia** — é filtro. A audiência é gerada só para a estratégia escolhida; o card de
analytics compara as três, sempre dentro das mesmas restrições.

## Scope

**In scope**

- Export offline (`simulator/export.py`) de **dois artefatos**: a **matriz de serving**
  pontuada (clientes × ofertas ativas, sem rótulo — gera a audiência) e o **holdout
  rotulado** (converted/lucro + scores + quadrante — alimenta o analytics no browser).
- HTML estático com a lógica de alocação **em JavaScript, espelhando
  `serve.recommend`** (melhor oferta por cliente → rank → corte por budget) e o cálculo
  de ganho incremental espelhando `gaincurve.incremental_gain_curve`.
- Uma **estratégia** escolhida (aleatório / conversão / uplift) + **filtros marcáveis**
  (tipo de campanha, quadrante de cliente) + slider de temperatura + botão **Gerar**.
- **Painel de lucro projetado** da audiência gerada (τ × lucro médio esperado por
  conversão tratada) — número em R$ na tela.
- **Card de analytics comparativo**: ao Gerar, compara a performance (curva de ganho
  incremental medida no holdout) das **três** estratégias, recomputada no browser sob
  os filtros marcados.
- **Download da campanha** (CSV) da estratégia escolhida.
- **Identidade visual iFood** e design com atenção a UX (REQ-312).
- **Seção "Sobre"/documentação para público não-técnico** explicando cada modo na
  prática (REQ-313).

**Out of scope**

- Backend/API em tempo de request; retreino no simulador.
- Score de modelo para `informational` (fora da modelagem por decisão registrada — ver
  REQ-303).
- Ação "não enviar" por cliente como política formal (removida com `policy.py`); o
  budget < N clientes já implica não enviar ao resto.
- Testes automatizados de UI (smoke de export + paridade Python cobrem o que quebra em
  silêncio).
- Segmentação K-Means como filtro de alvo — ver seção **Adicional** ao final; não
  bloqueia o MVP.
- Qualquer chamada de rede em tempo de uso além de `fetch` dos JSONs estáticos do
  próprio site (sem API externa, sem CDN de terceiros para dados).

**Non-goals**

- Não confundir lucro **projetado** (τ × lucro médio, forward) com ganho **medido**
  (contrafactual do holdout) — os dois aparecem, sempre rotulados.
- Não reimplementar features/scores fora de `src/`.
- Não depender de servidor próprio, banco de dados ou processo em background — tudo
  que a UI usa em runtime é arquivo estático servido pelo GitHub Pages.

## Público-alvo

**O usuário da interface é não-técnico** — um avaliador de negócio, não um cientista
de dados. Isso é uma restrição de produto, não um detalhe: toda a redação da UI, os
nomes dos controles, os tooltips e a documentação embutida SHALL ser escritos para quem
nunca ouviu "uplift", "X-learner", "Qini" ou "propensity". Termos técnicos, quando
inevitáveis, vêm sempre com uma explicação em linguagem comum ao lado. O jargão do
projeto (τ, CATE, contrafactual) fica na spec e no código — nunca cru na tela.

## Users & stories

- Como avaliador **não-técnico**, quero entender o que cada modo faz "na prática"
  (a quem manda cupom e por quê) sem precisar de fundamento estatístico.
- Como avaliador, quero escolher a estratégia, marcar restrições e clicar em **Gerar**
  para ver a audiência e o lucro projetado daquela campanha.
- Como avaliador, quero um **relatório comparando as três estratégias** sob as mesmas
  restrições, para justificar por que a proposta ganha — na mesma régua (curva de
  ganho incremental).
- Como avaliador, quero **baixar o CSV** da campanha que escolhi, para levar adiante.
- Como avaliador, quero uma seção "Sobre" que explique de onde vêm os números e o
  que confiar (projeção) vs. o que já foi medido (analytics no holdout).
- Como autor, quero regenerar tudo com um comando após `train`.

---

## Arquitetura

```
uv run python -m src.cli train                 (pré-requisito)
uv run python -m simulator.export               (job único, Spark + modelos)
        │
        ▼
simulator/data/
  metadata.json          # dimensões, defaults, lucro médio por conversão tratada, budgets do analytics
  offers.json            # catálogo (id, tipo, discount_value, min_value, duration)
  matrix.json            # matriz de SERVING pontuada, COLUNAR — gera a audiência (ver REQ-305)
  holdout.json           # holdout ROTULADO, COLUNAR — alimenta o analytics no browser (ver REQ-305)
        │
        ▼
simulator/index.html    # estático; alocação E cálculo de ganho rodam em JS
simulator/README.md     # passo a passo para o revisor
```

`simulator/` importa livremente de `src/` (é código Python normal rodando offline no
export) mas não é importado por `src/` — é uma camada de apresentação por cima do
projeto, não uma etapa do pipeline.

**Por que a alocação E o analytics rodam no browser:** estratégia, filtros e temperatura
são combináveis, e o analytics compara as três estratégias **sob os filtros que o
usuário marcou** — pré-computar toda combinação (estratégia × filtros) é combinatório.
Ordenar ~17k clientes e recomputar três curvas de ganho em JS é instantâneo. O export
congela a parte cara (Spark + modelos): os **scores** por linha e o **holdout rotulado**;
o browser refaz a parte barata — a **seleção** (`serve.recommend`, ~12 linhas) e o
**ganho incremental** (`gaincurve.incremental_gain_curve`, contrafactual escalado estilo
Qini). Nenhuma fórmula nova: as duas são espelhos exatos de `src/`, guardados por teste.

---

## Functional requirements

### REQ-301 — Export da matriz pontuada

WHEN `simulator.export` roda com modelo treinado, the system SHALL:

1. Montar `serve.build_scoring_frame(spark, cfg, decision_time, active_offer_ids)` com
   **todas** as ofertas do catálogo (bogo, discount, informational) — o filtro de tipo é
   da UI, não do export.
2. Pontuar as linhas bogo/discount com os componentes do `BlendedUpliftModel`
   carregado: `uplift` (τ), `p_convert`, `uncertainty`, e o score do blend dinâmico γ
   (`gaincurve.dynamic_hybrid_score`). Linhas `informational` ficam com scores nulos (o
   modelo não as conhece).
3. Classificar quadrante por linha: `quadrant.classify_quadrant(stages, p_convert,
   cfg)` sobre `UpliftModel.predict_stages`.
4. Calcular, sobre o holdout, `lucro_medio_por_conversao_tratada`
   (`gaincurve._profit_per_treated_conversion` no budget máximo disponível) e gravá-lo
   em `metadata.json` — é o insumo único do lucro projetado (REQ-307).

**Acceptance**

- GIVEN N clientes e M ofertas THEN a matriz tem N×M linhas; as `informational`
  presentes com `uplift=null`.
- GIVEN duas execuções com mesma config/seed THEN artefatos idênticos byte a byte.

### REQ-302 — Estratégias delegadas (zero fórmula nova)

The UI SHALL oferecer **três estratégias** de ranqueamento — e **só três**. Alvo e
segmento **não são estratégias**: são filtros (REQ-306), aplicáveis a qualquer uma
delas. Cada estratégia vem de função existente:

| Estratégia | Fonte |
|---|---|
| `aleatorio` | sem score; ordem por seed no browser (equivalente a `gaincurve.random_ranking`) |
| `conversao` | `ConversionModel.predict_proba` (o baseline "premiar" que a curva de ganho já compara) |
| `uplift` | `BlendedUpliftModel` **modo dinâmico γ=1,0** (`gaincurve.dynamic_hybrid_score`) — o melhor do holdout |

τ puro (`UpliftModel.predict`) também é exportado como coluna, para o lucro projetado e
para diagnóstico, mas a estratégia "uplift" da UI é o blend dinâmico (decisão: é o
modelo de produção, não o X-learner cru).

### REQ-303 — Filtro por tipo de campanha

The UI SHALL exibir checkboxes por `offer_type`: **bogo ☑ · discount ☑ ·
informational ☐** (desmarcada por default).

- Marcar `informational` só tem efeito sob score-base `aleatorio` (única estratégia que
  não precisa de score de modelo); sob `conversao`/`uplift` as linhas informational são
  excluídas do argmax com aviso visível: *"informational está fora da modelagem — sem
  score; disponível apenas no envio aleatório"*.
- O filtro age **antes** do "melhor oferta por cliente": desmarcar um tipo redistribui
  clientes para a melhor oferta restante.
- Linhas informational não contribuem para o lucro projetado (τ nulo; sem
  `discount_value` por construção).

**Acceptance**

- GIVEN só `discount` marcado THEN nenhuma linha bogo/informational aparece na
  alocação e cada cliente recebe sua melhor oferta discount.

### REQ-304 — Alocação da audiência no browser = `serve.recommend` em JS

WHEN o usuário clica em **Gerar**, the UI SHALL montar a audiência da **estratégia
escolhida** aplicando, nesta ordem (a alocação é disparada por Gerar, ação explícita —
não a cada tecla):

1. **Filtro de ofertas** (REQ-303) e **filtros de alvo** (REQ-306) sobre a matriz.
2. **Melhor oferta por cliente**: argmax do score da estratégia por `account_id`,
   desempate estável por `offer_id`.
3. **Rank por temperatura**: `temperature=0` → ordenação determinística por score;
   `>0` → Gumbel-max sobre score min-max normalizado (a mesma matemática de
   `gaincurve.softmax_ranking`), com PRNG seedado (seed no metadata) para
   reprodutibilidade.
4. **Corte**: primeiros `budget` clientes.

Os mesmos filtros (passo 1) são aplicados ao holdout do analytics (REQ-308), para a
comparação medir cada estratégia sobre a **mesma** população restrita da audiência.

**Acceptance (paridade)**

- GIVEN estratégia uplift, filtros default, `temperature=0`, budget B THEN os B
  primeiros `(account_id, offer_id)` são **idênticos** aos de
  `serve.recommend(scored, B, temperature=0)` em Python sobre os mesmos artefatos —
  guardado por smoke test que exporta uma fixture pequena e compara com a saída JS de
  referência (ou reimplementação do algoritmo JS em Python no teste).
- Paridade estocástica (τ>0) **não** é exigida entre linguagens; exige-se apenas
  reprodutibilidade dentro do browser dado o mesmo seed.

### REQ-305 — Contrato dos artefatos

Dois artefatos de dados, ambos em formato **colunar** (arrays paralelos, não array de
objetos), com papéis distintos:

`matrix.json` — **serving**, gera a audiência (uma linha por par cliente × oferta ativa
as-of a decisão, sem rótulo):

```json
{
  "account_id": [...], "offer_id": [...], "offer_type": [...],
  "uplift": [...], "p_convert": [...], "score_dynamic": [...],
  "quadrant": [...]
}
```

`holdout.json` — **holdout rotulado**, alimenta o analytics (uma linha por linha do
holdout de avaliação; tem rótulo e braço, que a matriz de serving não tem):

```json
{
  "account_id": [...], "offer_type": [...],
  "treatment": [...], "converted": [...], "net_profit": [...],
  "score_random": [...], "p_convert": [...], "score_dynamic": [...],
  "quadrant": [...]
}
```

- `net_profit` sai de `gaincurve.add_net_profit` (`conversion_value − reward_cost` por
  linha) — o mesmo insumo da curva de ganho em `src/`.
- `score_random` é o score aleatório seedado (para o browser reproduzir
  `gaincurve.random_ranking`); `p_convert` e `score_dynamic` são as estratégias
  conversão e uplift. Assim o browser ranqueia as três **sem** reajustar modelo.
- `quadrant` é o mesmo do serving, para os filtros de alvo agirem no holdout.

`metadata.json`: `decision_time`, `n_clients`, `seed`,
`lucro_medio_por_conversao_tratada` (holdout, insumo do REQ-307), `analytics_budgets`
(de `cfg.gain_curve_budgets` — os pontos do card de comparação), defaults dos controles,
texto do disclaimer, labels/descrições das estratégias para os cards.

**Acceptance:** HTML carrega tudo via `fetch` relativo (`./data/*.json`), sem CORS,
sem servidor próprio; tamanho de cada arquivo committed **< 50 MB** (limite de warning
do GitHub) e o total de `simulator/data/` **< 200 MB** (folga generosa sob o soft cap
de ~1 GB por repositório do GitHub) — formato colunar já reduz as linhas a poucos
arrays primitivos, tipicamente low-digit MB por coluna numérica.

### REQ-306 — Filtros de alvo (marcáveis, não estratégia)

Alvo é **filtro**, não estratégia: um conjunto de restrições de público que o usuário
**marca ou não**, componíveis com qualquer estratégia (REQ-302). A UI SHALL oferecê-los
como campos marcáveis em linguagem leiga:

| Filtro (checkbox) | Implementação |
|---|---|
| "Só quem o cupom convence" (persuadables) | mantém só linhas com `quadrant == persuadable`; por construção exclui sleeping dogs |
| "Excluir quem compraria sem cupom" (sem sleeping dogs) | remove só `quadrant == sleeping_dog` (versão branda, público maior) |

Nenhum marcado = público inteiro. Marcar os dois compõe (intersecção das restrições).
Filtro por segmento K-Means é **mais um campo marcável** quando entrar — ver seção
**Adicional**; não faz parte do core do MVP.

Os filtros valem **igualmente** para a audiência (REQ-304) e para todas as estratégias
do analytics (REQ-308) — "públicos de cada estratégia dentro das restrições propostas".
O quadrante vem **pré-computado no export** (REQ-301), presente tanto na matriz de
serving quanto no `holdout.json`; o browser só aplica máscara, nunca reclassifica.

**Acceptance:** GIVEN "só persuadables" marcado THEN a audiência e cada estratégia do
analytics são medidas apenas sobre linhas persuadable; GIVEN público filtrado menor que
o budget THEN a UI reporta "público esgotado: K < N" em vez de completar com
fora-do-alvo.

### REQ-307 — Lucro projetado (em escopo, com rótulo)

Para o top-N alocado, the UI SHALL exibir:

- **Conversões incrementais esperadas** = Σ τᵢ (linhas com τ; informational fica
  fora).
- **Lucro incremental projetado** = Σ τᵢ × `lucro_medio_por_conversao_tratada`, onde o
  segundo termo é o número **único** do `metadata.json` (REQ-301 item 4) — a mesma
  grandeza estável (~R$14–20 no holdout) usada na fatoração de
  `gaincurve.incremental_gain_curve` (`lucro_incremental(N) = conversao_incremental(N)
  · lucro_medio_por_conversao_tratada(N)`). Não há ticket por cliente nem fallback: o
  lucro projetado espelha a mesma fatoração de §5, só que com τ **previsto** por linha
  no lugar do contrafactual medido por prefixo.
- **Orçamento de desconto projetado** = Σ p_convertᵢ × `discount_value`(ofertaᵢ) — linha
  **informativa**, não subtraída do lucro. O `lucro_medio_por_conversao_tratada` já é
  líquido (`conversion_value − reward_cost` por linha, ver `gaincurve.add_net_profit`);
  descontar o orçamento de novo do total dobraria a contabilização do custo. O painel
  mostra as duas linhas lado a lado, rotuladas separadamente ("lucro líquido projetado"
  vs. "orçamento de desconto esperado"), sem somar uma na outra.
- **Lucro incremental projetado** é o número final exibido como "ROI" — não há dedução
  adicional.

Rótulo obrigatório junto ao número: **"Projeção (τ × lucro médio por conversão) — a
medição causal está no card de analytics"**. A distinção entre projeção e medição é
requisito, não nota de rodapé: este cálculo é da mesma família do
`expected_net_profit` removido com `policy.py`; ele volta aqui **como leitura de
simulador**, não como critério de alocação (o ranking nunca usa o lucro projetado —
usa o score da estratégia escolhida).

**Acceptance:** GIVEN budget maior THEN conversões esperadas são não-decrescentes;
GIVEN estratégia conversão vs uplift no mesmo budget THEN os painéis divergem de forma
explicável (conversão: orçamento de desconto maior, τ menor); GIVEN uma fixture
pequena THEN o lucro projetado bate com a soma manual `Σ τᵢ ×
lucro_medio_por_conversao_tratada`.

### REQ-308 — Card de analytics comparativo (recomputado no browser sob os filtros)

WHEN o usuário clica em **Gerar**, the UI SHALL recomputar, **no browser**, a
performance das **três** estratégias sobre o `holdout.json` filtrado pelos mesmos
filtros marcados (REQ-306), e exibir a comparação num card de analytics.

Para cada estratégia (aleatório/conversão/uplift):

1. filtra o holdout pelos filtros de alvo/oferta marcados (a mesma população da
   audiência);
2. ranqueia as linhas pela coluna de score da estratégia (`score_random`/`p_convert`/
   `score_dynamic`);
3. calcula a **curva de ganho incremental** nos `analytics_budgets` do metadata,
   espelhando `gaincurve.incremental_gain_curve` — contrafactual escalado estilo Qini
   sobre `net_profit`/`converted`. Nenhuma fórmula nova: é o espelho JS de
   `src/gaincurve.py`, com paridade guardada por teste (REQ-310).

O card compara as três lado a lado (tabela e/ou curva no tema de figuras) — é a régua
única da banca (status quo → premiar → uplift), agora **sensível aos filtros**: mudar as
restrições e Gerar de novo recompõe a comparação sobre o novo público.

**Rótulo obrigatório:** este é o ganho **medido no holdout** (o que já aconteceu),
distinto do **lucro projetado** da audiência (REQ-307, uma estimativa para frente) — os
dois convivem na tela, sempre rotulados (ver REQ-313).

**Acceptance:** GIVEN filtros default e Gerar THEN o card mostra as três estratégias nos
budgets do metadata, com uplift concentrando mais ganho incremental; GIVEN "só
persuadables" marcado e Gerar THEN os números das três estratégias mudam (medidos só
sobre persuadables); GIVEN uma fixture pequena THEN a curva de cada estratégia bate com
`gaincurve.incremental_gain_curve` em Python (REQ-310).

### REQ-309 — Configuração (REQ-110)

```yaml
simulator_output_dir: "simulator/data"
simulator_default_budget: 1000
simulator_score_gamma: 1.0            # espelha blend_gamma (score-base uplift)
simulator_temperature_default: 0.0    # slider começa em exploit puro
simulator_temperature_max: 1.0        # teto do slider
```

Nada hardcoded no export nem no JS (defaults do JS vêm de `metadata.json`).

### REQ-310 — Entrypoint, docs e smoke

- `uv run python -m simulator.export [--config ...]`; `simulator/README.md` com os 4
  passos (pipeline → train → export → `python -m http.server` em `simulator/` **para
  teste local** — em produção o servidor é o próprio GitHub Pages).
- Smoke test Python (`tests/test_simulator_export.py`): uma oferta por cliente,
  paridade determinística com `serve.recommend` (REQ-304), **paridade da curva de ganho
  por estratégia com `gaincurve.incremental_gain_curve`** (REQ-308) numa fixture, lucro
  projetado do REQ-307 bate com a soma manual, linhas informational sem score nunca
  entram no argmax de estratégia modelada.

### REQ-311 — Compatibilidade com GitHub Pages

WHEN o simulador é publicado, the system SHALL rodar inteiramente como site estático,
sem nenhuma dependência que exija servidor próprio:

- `simulator/index.html` usa apenas HTML/CSS/JS puro (ou bibliotecas via `<script>`
  vendorizado/CDN público, nunca módulo Node com build step) — GitHub Pages serve
  arquivos como estão, sem transpilar nem empacotar.
- Todos os `fetch` são para caminhos **relativos** dentro do próprio site
  (`./data/matrix.json` etc.), nunca `localhost` ou caminho absoluto de disco — para
  funcionar tanto em `usuario.github.io/repo/simulator/` (project page, com prefixo de
  path) quanto em teste local.
- Artefatos de `simulator/data/` são **committados no repositório** (não gerados em
  CI nem baixados de storage externo) — GitHub Pages não roda build arbitrário; o
  `simulator.export` é executado localmente pelo autor antes do commit, e o resultado
  é o que fica publicado.
- Sem cookies, sessão, autenticação ou estado no servidor — todo o estado (knobs,
  seed) vive na URL/localStorage do browser, opcional e não persistido entre
  visitantes.

**Acceptance:**
- GIVEN o conteúdo de `simulator/` publicado como GitHub Pages (root ou subpasta)
  THEN a página carrega, faz fetch dos JSONs e permite interagir com os knobs sem
  nenhum erro de console relacionado a path absoluto ou CORS.
- GIVEN o repositório com `simulator/data/` committado THEN nenhum arquivo individual
  ultrapassa 50 MB (ver REQ-305) e o `git push` não é bloqueado pelo GitHub.

### REQ-312 — Identidade visual e design

WHEN a interface é renderizada, the system SHALL apresentar uma identidade visual
alinhada ao iFood — elegante e agradável, com atenção a UX, sem excesso de efeitos.

- **Cor de marca:** o chrome da interface (cabeçalho, botões primários, destaques de
  seleção, barras de ênfase) usa o **vermelho iFood** (`#EA1D2C` como primária, com
  tons de apoio derivados) sobre superfícies neutras claras. A cor de marca é para a
  **interface**; as figuras embutidas (curva de ganho, composição) continuam com a
  paleta CVD-validada de `src/viz.py` (`ifood_light`/`ifood_dark`) — os dois propósitos
  são distintos (marca vs. legibilidade estatística) e não se misturam: nunca pintar
  uma série de gráfico de vermelho iFood só por identidade.
- **Tipografia:** fonte sem serifa próxima à da marca — a família do próprio `viz.py`
  serve de âncora (`Inter, Helvetica, Arial, sans-serif`); se uma face mais próxima do
  iFood estiver disponível como webfont **autohospedada** (sem CDN de terceiros em
  runtime, para não quebrar o REQ-311/GitHub Pages), pode ser usada com fallback para
  Inter. Hierarquia tipográfica clara (título, seção, corpo, legenda).
- **Layout e UX:** espaçamento generoso, hierarquia visual óbvia (o número de lucro, a
  lista e o card de analytics são o foco; os controles são secundários mas acessíveis),
  estados de foco/hover visíveis, contraste adequado (WCAG AA para texto), e o botão
  **Gerar** com destaque como a ação principal. Nada de animação gratuita — transições
  sutis só onde ajudam a entender o que mudou.
- **Responsivo o suficiente** para ser apresentado num laptop e num projetor; não
  precisa de suporte a mobile fino, mas não pode quebrar horizontalmente.
- **Tema:** claro por padrão (identidade iFood é clara); modo escuro é opcional e
  não-bloqueante.

**Acceptance:**
- GIVEN a página aberta THEN a identidade iFood é reconhecível (vermelho de marca no
  chrome, tipografia coerente) sem que nenhuma figura estatística perca a paleta
  validada de `viz.py`.
- GIVEN qualquer webfont THEN ela é autohospedada em `simulator/` (nenhum fetch a
  `fonts.googleapis.com` ou CDN externo em runtime — REQ-311).

### REQ-313 — Seção "Sobre" e documentação para não-técnicos

The UI SHALL conter uma seção **"Sobre"** (About) e documentação embutida que explique,
**em linguagem para público não-técnico**, como o simulador e cada modo funcionam na
prática.

- **O que é isto:** um parágrafo curto explicando que o simulador mostra, para um
  orçamento de N cupons, a quem enviar e quanto isso deve render — e que ele apenas
  *ordena e conta* sobre dados já coletados, não roda campanha real.
- **Como cada modo funciona na prática**, um bloco por modo, sem jargão:
  - *Aleatório (status quo):* "manda cupom para quem calhar — é o ponto de partida
    para comparar."
  - *Premiar quem já compra (CRM de hoje):* "manda para quem tem mais chance de
    comprar — mas essas pessoas talvez comprassem de qualquer jeito, então parte do
    desconto é dinheiro gasto à toa."
  - *Maximizar uplift (a proposta):* "manda para quem só compra **por causa** do
    cupom — o modelo aprende a diferença entre quem foi convencido e quem já ia
    comprar." Explicar a intuição de "efeito incremental" com um exemplo concreto,
    sem a palavra "uplift" antes de defini-la.
  - *Filtros (alvo e exploração):* explicar que os filtros **não** são um modo à parte —
    são restrições que se marcam por cima de qualquer estratégia (mandar só para certos
    tipos de cliente), e a exploração (temperatura) mistura um pouco de acaso para não
    ficar preso sempre nos mesmos.
- **Fluxo Gerar:** explicar que se escolhe a estratégia, marcam-se os filtros e clica-se
  em Gerar; então aparece a lista da campanha e o relatório comparando as três
  estratégias sob as mesmas restrições.
- **Projeção vs. medição:** explicar a diferença entre o número de lucro **projetado**
  (uma estimativa para frente, "o que esperamos" da campanha escolhida) e o **analytics
  no holdout** (o que já aconteceu nos dados reais, comparando as estratégias) — por que
  os dois existem e em qual confiar para quê.
- **Os quadrantes** (persuadables, sleeping dogs etc.) explicados como "tipos de
  cliente" em português claro, para dar sentido aos filtros de alvo.
- **De onde vêm os dados e limites conhecidos:** uma nota honesta de que é uma
  avaliação offline sobre histórico, não um teste A/B ao vivo, e que "quem viu o
  cupom" foi escolha do cliente (confundimento residual) — dito em uma frase simples.

Conteúdo estático (não depende de dados carregados); pode viver numa aba/seção
dedicada ou num painel expansível, desde que descobrível.

**Acceptance:**
- GIVEN um avaliador sem formação técnica THEN a seção "Sobre" permite entender o que
  cada modo faz e a diferença projeção/analytics sem consultar ninguém.
- GIVEN a documentação THEN nenhum termo técnico aparece sem explicação em linguagem
  comum na mesma tela.

### REQ-314 — Download da campanha escolhida

WHEN o usuário clica em "Baixar campanha (CSV)", the UI SHALL exportar a **audiência da
estratégia escolhida** (a lista gerada em REQ-304), gerada no browser (Blob de texto),
sem backend.

- Colunas = `serve.RECOMMENDATION_COLUMNS` (`rank`, `account_id`, `offer_id`,
  `offer_type`, `score`), na ordem da seleção.
- A campanha baixada respeita o budget, os filtros e a temperatura vigentes no momento
  do Gerar que a produziu — é exatamente a lista exibida, não uma recomputação.
- O botão só fica disponível **após Gerar**; antes disso não há audiência a baixar.
- É a **estratégia escolhida** que se baixa (uma campanha), não uma por estratégia — o
  analytics compara as três, mas a campanha operacional é uma só.

**Acceptance:** GIVEN uma audiência gerada THEN o CSV baixado tem exatamente as linhas
exibidas, nas colunas de `RECOMMENDATION_COLUMNS`, na mesma ordem de `rank`; GIVEN nada
gerado ainda THEN não há download disponível.

---

## Os 5 itens da banca, mapeados em estratégia + filtros

As três primeiras histórias são **estratégias**; as duas últimas são **refinamentos**
(filtros/temperatura) que se aplicam por cima de qualquer estratégia — não são modos à
parte:

| História | Estratégia + filtros |
|---|---|
| 1. Aleatório (status quo) | estratégia `aleatorio` · sem filtro · τ=0 |
| 2. Premiar (CRM hoje) | estratégia `conversao` · sem filtro · τ=0 |
| 3. Maximizar uplift (a proposta) | estratégia `uplift` (blend γ=1) · sem filtro · τ=0 |
| 4. Alvo (refinamento) | qualquer estratégia · filtro `persuadables` e/ou `sem sleeping dogs` |
| 5. Exploração (refinamento) | qualquer estratégia · temperatura τ > 0 |

A UI pode oferecer as três estratégias como **presets** (um clique escolhe a estratégia
e limpa filtros), mantendo estratégia, filtros e temperatura visíveis para o avaliador
mexer — presets contam a história, os controles mostram a engenharia. O card de
analytics compara sempre as três, para a história 3 se provar contra 1 e 2 na mesma
tela, sob quaisquer filtros das histórias 4–5.

## Definition of Done

- [ ] Export roda end-to-end no dado real após `train`; `matrix.json` + `holdout.json`
  reproduzíveis por seed.
- [ ] UI: escolher estratégia + marcar filtros (oferta + quadrante) + budget +
  temperatura + **Gerar** funcionando sem backend.
- [ ] Lucro projetado (Σ τ × lucro médio por conversão tratada) em R$ na tela, com
  rótulo de projeção, para a estratégia escolhida.
- [ ] Card de analytics comparando as três estratégias, recomputado no browser sob os
  filtros marcados (REQ-308).
- [ ] Download da campanha escolhida em CSV (REQ-314).
- [ ] Paridade determinística com `serve.recommend` **e** com
  `gaincurve.incremental_gain_curve` guardada por teste.
- [ ] "Público esgotado" tratado; informational só entra no aleatório.
- [ ] Config completa (REQ-309); `simulator/README.md` para o revisor.
- [ ] Publicado como GitHub Pages funcionando (REQ-311): paths relativos, sem
  backend, artefatos committados dentro do limite de tamanho.
- [ ] Identidade visual iFood aplicada, design agradável e legível (REQ-312).
- [ ] Seção "Sobre" explica o fluxo Gerar e cada modo na prática para público
  não-técnico; nenhum termo técnico sem tradução na tela (REQ-313).

## Riscos registrados

1. **Lucro projetado vs. decisão passada:** o repo removeu `policy.py` por tratar
   τ×receita como alocação; aqui o lucro é *leitura*, nunca critério de ranking — o
   teste do REQ-310 e o rótulo do REQ-307 guardam essa fronteira.
2. **Lucro médio é uma média global aplicada a um top-N selecionado:** o
   `lucro_medio_por_conversao_tratada` do metadata é medido sobre o holdout inteiro
   (ou no budget máximo); a composição real de um prefixo pequeno pode ter
   lucro/conversão por evento diferente da média global. O card de analytics (medição
   causal por budget, REQ-308) é o contrapeso na mesma tela.
3. **Softmax JS ≠ NumPy:** paridade estocástica entre linguagens não é meta;
   determinística é.
4. **Cálculo de ganho em JS pode divergir de `gaincurve.py`:** o analytics reimplementa
   o contrafactual escalado estilo Qini no browser; um desvio silencioso mediria as
   estratégias errado. O teste de paridade do REQ-310 (curva JS vs.
   `gaincurve.incremental_gain_curve` numa fixture) é a trava.

---

## Adicional (se sobrar tempo): filtro por segmento K-Means

Fora do core do MVP — implementar só se os REQ-301…310 estiverem prontos e sobrar
tempo.

- **Export**: anexar a coluna de segmento K-Means por cliente
  (`eda.cluster_matrix → fit_clusters → assign_segments`, k da varredura por
  silhouette), incluindo o segmento sentinela de `identity_missing`, **tanto na
  `matrix.json` quanto no `holdout.json`** (o filtro precisa agir nos dois, como o
  quadrante). Opcionalmente `simulator/data/clients.json` para rótulos legíveis dos
  segmentos.
- **UI**: acrescentar **mais um campo marcável** aos filtros de alvo (REQ-306) — um
  select de segmento — que restringe o público ao segmento escolhido, compondo com os
  filtros de quadrante e valendo igualmente para a audiência e para o analytics.
- **Risco a registrar quando implementado:** segmento K-Means usa a janela inteira do
  histórico do cliente — aceitável como filtro de persona no serving (não é feature de
  modelo; o leakage proibido era no treino, não no filtro de exibição).
