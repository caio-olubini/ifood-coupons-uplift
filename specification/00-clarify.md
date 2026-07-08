# 00 — Clarify: fundação do projeto

> Status: Ready · Fase 0 (Clarify) do SDD
> Escopo: decisões, premissas e limites que valem para todas as specs abaixo.

## Objetivo do projeto

Decidir, para cada cliente, **qual oferta enviar (ou não enviar)** de modo a maximizar
lucro líquido, e **provar o ganho da política sem rodar o A/B**, via avaliação offline
sobre os dados históricos.

A pergunta não é "quem vai completar a oferta?" (classificação) — é "para quem a oferta
**muda** o comportamento?" (incrementalidade / uplift).

## Escopo

**Dentro**
- Pipeline PySpark local: limpeza, construção de label influence-aware, atribuição
  temporal oferta→transação, features pré-recebimento, dataset unificado.
- EDA que permita avaliar os dados e diagnosticar o experimento (incl. balanço de covariáveis).
- Modelo de uplift (X-learner) com contrato de saída (uplift por cliente × oferta).
- Política de alocação multi-tratamento sensível a custo, incluindo a ação "não enviar".
- Avaliação offline por IPW comparando a política contra baselines de campanha.
- Impacto em R$ e apresentação executiva (≤5 slides).

**Fora**
- A/B online (é o próximo passo dimensionado, não uma entrega).
- Deep learning (17k clientes tabulares — árvores vencem).
- Databricks / execução em nuvem (rodamos local via UV).
- Personalização de texto/canal da mensagem (LLM) — citado como próximo passo, não implementado.

**Não-objetivos** (mata a suposição na origem)
- Não otimizamos taxa de completion. Otimizar completion premia canibalização.
- Não estimamos propensity score para de-confounding: os dados são de um experimento
  aleatorizado; a propensity é constante conhecida (ver Premissa 4).
- Não validamos linha a linha com Pydantic dentro do Spark (ver Premissa 7).

## Premissas numeradas

Cada premissa é uma escolha consciente. Onde há risco, a limitação está nomeada e o
upgrade correspondente listado.

1. **Uma oferta ativa por vez, por cliente.** A atribuição de uma transação a uma oferta
   assume que, na janela de validade, há no máximo uma oferta ativa para aquele cliente.
   Simplifica a atribuição temporal. *Limitação:* janelas sobrepostas existem no dataset;
   o upgrade é atribuição por sobreposição (regra de prioridade). Documentado, não silencioso.

2. **Conversão é influence-aware.** Uma oferta só conta como convertida se foi
   **vista E completada dentro da duração**. Completado sem view precedente = compra que
   aconteceria de qualquer forma (sure thing), não conversão causada. Base empírica:
   28,4% dos pares (cliente, oferta) com "completed" têm completed sem view antes.

3. **Idade 118 é sentinela de identidade ausente, não idade real.** Os 2.175 clientes com
   `age=118` são exatamente os mesmos sem `gender` e sem `credit_card_limit` (sobreposição
   perfeita, 12,8% da base). Tratados como segmento `identity_missing`, nunca descartados,
   nunca imputados para dentro de estatísticas de idade.

4. **Os dados vêm de um experimento aleatorizado (RCT).** A atribuição de ofertas é
   uniforme por desenho. Consequência: a propensity de tratamento é **constante e conhecida**,
   não estimada. O IPW simplifica (peso fixo). O check de balanço de covariáveis **verifica**
   essa premissa em vez de assumi-la — é diagnóstico, não gate.

5. **Modelo de uplift: X-learner (único).** Escolhido por robustez a grupos de tamanhos
   desiguais e a μ₀ mal-estimado — a fraqueza exata do T-learner. Decisão fixa, sem bifurcação.

6. **Baselines de política são de campanha, não de modelo.** A avaliação offline compara a
   política de uplift contra: (a) aleatória, (b) enviar-a-todos [status quo que gerou os dados],
   (c) top-completion [a política que um modelo de classificação produziria]. Ver Premissa 8
   para o limite do método.

7. **Pydantic vive nas bordas, não no caminho quente do Spark.** Validação em massa é
   trabalho de schema Spark + asserções sobre o DataFrame. Pydantic valida config,
   o contrato de schema (em amostra/schema, não por-linha) e artefatos de saída.
   Validar linha a linha dentro de UDF destruiria a performance — proibido.

8. **IPW só avalia políticas com sobreposição no observado (positividade).**
   "Enviar-a-todos" e "aleatória" têm sobreposição total. "Não enviar a ninguém" **não é
   avaliável** por IPW puro (não há no dataset clientes que não receberam nada). Limitação
   nomeada; a política pode *recomendar* não enviar, mas o valor absoluto dessa ação não é
   estimável offline sem premissa adicional.

## Configurabilidade (princípio transversal)

Nenhum parâmetro de comportamento é hardcoded. Janelas de atribuição, limiar de balanço
(SMD), custos por tipo de oferta, cortes temporais de validação, caminhos e seeds vivem
num objeto de configuração tipado e validado (Pydantic), carregado no início de cada etapa.
Um valor mágico dentro de uma função é um defeito, não um detalhe.

## Perguntas ao recrutador

Nenhuma bloqueante. As dúvidas de dados foram resolvidas por inspeção e viraram premissas
acima. Se alguma premissa for rejeitada pela banca, o upgrade correspondente já está nomeado.

## Critérios de sucesso do projeto

- Pipeline reproduzível por CLI, do dado bruto ao dataset unificado, com testes de
  integridade passando.
- Um número de lucro líquido em R$ na tela, com a política de uplift superando os três
  baselines na avaliação offline.
- Um leitor técnico consegue, pela EDA, entender o que há nos dados e por que a modelagem
  foi enquadrada como uplift.
- Um líder de negócio leigo entende o ganho pelos slides sem jargão.
