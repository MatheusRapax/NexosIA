# Relatório Técnico — Developmental PCN

> Registro de engenharia do processo que produziu este repositório: como foi conduzido, o que foi construído, os bugs e decisões de design encontrados pelo caminho, e um guia prático para quem for continuar o trabalho.

## 1. Processo: fluxo Arquiteto/Executor

O projeto foi conduzido por dois agentes de IA conectados via Maestri, com papéis fixos:

- **Arquiteto** (Claude Code): nunca escreveu código de implementação. Escreveu specs completas (objetivo, algoritmo exato quando havia risco de ambiguidade, interfaces obrigatórias, critérios de aceite mensuráveis, arquivos a tocar, constraints, escopo explícito) numa nota compartilhada, delegou, e **verificou cada entrega de forma independente** — lendo o código real, executando os testes e experimentos ele mesmo, nunca confiando apenas no resumo reportado pelo Executor.
- **Executor** (Antigravity): implementou exatamente o especificado, rodou os testes, e reportou um resumo estruturado (resultado numérico, confirmação de constraints, desvios da spec com justificativa, status final).

Esse processo se mostrou valioso de duas formas concretas:
1. Em várias ocasiões, a verificação independente do Arquiteto encontrou problemas que o resumo do Executor não mencionava (ex: um teste com tolerância afrouxada sem disclosure na Fase 1; uma mudança em arquivo "congelado" sem disclosure na Fase 5).
2. Em outras ocasiões, o Executor se recusou corretamente a mascarar um resultado ruim ou inventar um mecanismo não autorizado para "passar no teste" — reportando BLOQUEADO com diagnóstico técnico em vez de forçar um APROVADO falso (Fase 4, tentativa 1; Fase 5, todas as 4 tentativas).

## 2. Inventário de arquivos por fase

```
pcn_core/
  model.py            — PredictiveCodingNetwork (motor local, Fase 1; usado sem alteração por todas as fases seguintes)
  sleep.py            — SleepConsolidator (Fase 4; alpha tornado configurável na Fase 5)
  data.py             — carregamento do dataset digits (Fase 1)
  baseline.py         — MLP com backprop, só para comparação (Fase 1)
  run_experiment.py   — script de demonstração da Fase 1
  tests/test_pcn_vs_backprop.py

active_inference/
  environment.py      — GridWorld 5x5 (Fase 2; nunca modificado depois)
  agent.py            — ActiveInferenceAgent (Fase 2: valor pragmático; Fase 3: + Learning Progress, no mesmo arquivo)
  random_baseline.py  — RandomAgent (Fase 2)
  learning_progress.py — LearningProgressTracker (Fase 3)
  run_phase2_experiment.py
  run_phase3_experiment.py
  tests/test_phase2_learning.py
  tests/test_phase3_learning_progress.py
  README.md

consolidation/
  run_phase4_experiment.py  — experimento sintético de esquecimento catastrófico
  tests/test_catastrophic_forgetting.py
  README.md

integration/            — FASE 5, BLOQUEADA (não aprovada, SleepConsolidator antigo)
  run_unified_agent.py
  tests/test_unified_agent.py
  README.md            — diagnóstico técnico do bloqueio definitivo

dual_weight/            — FASE 6, APROVADA (detector de changepoint Dual-Weight PCN)
  dual_weight_pcn.py    — classe DualWeightPCN (Fast/Slow Networks, Polyak averaging)
  tests/test_changepoint_detector.py
  README.md

integration_v2/          — FASE 7, BLOQUEADA (achado negativo quantificado: 25.09% de 30% exigido)
  run_integration_experiment.py
  tests/test_dual_weight_integration.py
  README.md            — trajetória completa das 3 tentativas (2.84% -> 17.79% -> 25.09%)

integration_v3/          — FASE 8, APROVADA (replay priorizado suavizado: 64.91% de reducao, caso de 2 regioes)
  run_integration_experiment.py   — sample_dream_cell_action com priorizacao suavizada (alpha=0.5, epsilon=1.0)
  tests/test_prioritized_sampling.py
  tests/test_dual_weight_integration_v3.py
  README.md            — compara as 2 tentativas desta fase (-11.06% -> 64.91%) e com a Fase 7

integration_v4/          — FASE 9, CONCLUIDA (exploratoria -- confirma previsao teorica com 3 regioes)
  run_integration_experiment.py   — mesmo mecanismo da Fase 8, snapshot atualizado a cada nova borda de subida
  tests/test_three_regions.py
  README.md            — A apos B=66.24%, A apos C=-17.23% (protecao perdida), B apos C=51.19% (tarefa recente protegida)

ewc_local/                — FASE 10, Mecanismo B (EWC Local Online)
  ewc_tracker.py        — classe EWCLocalTracker (F_ema, F_total, w_anchor por peso; update_ema/on_changepoint/apply_penalty)
  tests/test_ewc_tracker.py   — validacao isolada (4 casos: 1o changepoint, decaimento multi-changepoint, direcao da penalidade, F zerado)

integration_v5/          — FASE 10, CONCLUIDA-COM-RESSALVA (Mecanismo A+B juntos, calibracao de lambda)
  run_integration_experiment.py   — protocolo A->B->C da Fase 9 + EWCLocalTracker; main() varre lambda em {0.01,0.1,0.2,0.3,1.0}
  tests/test_mechanism_b.py   — roda o experimento reduzido (10 eps/regiao) de ponta a ponta com EWC ativo
  README.md            — tabela completa de calibracao; nenhum lambda resolveu o trade-off plasticidade-vs-estabilidade

pcn_core/tests/test_last_local_grad_equivalence.py  — FASE 10, equivalencia numerica apos a 1a modificacao historica em model.py

docs/
  DEBATE_ORIGINAL.md    — reconstrução do debate (Rodada 1) que originou a arquitetura
  DEBATE_RODADA2.md     — segunda rodada de debate (diagnóstico da Fase 5, mecanismo Dual-Weight PCN)
  DEBATE_RODADA3.md     — terceira rodada de debate (observação do usuário sobre atenção/importância, mecanismo de replay priorizado)
  ARTIGO.md             — artigo científico (resultados consolidados, Fases 1-10)
  RELATORIO_TECNICO.md  — este documento
```

## 3. Como reproduzir cada fase

Todos os comandos abaixo assumem Python 3.x com `numpy`, `torch`, `scikit-learn` e `pytest` instalados, executados da raiz do repositório.

```bash
# Fase 1 — motor PCN vs. backprop
python pcn_core/run_experiment.py
pytest pcn_core/tests/ -v

# (Para qualquer script que importe active_inference/agent.py diretamente de outra pasta,
#  defina PYTHONPATH incluindo a raiz do projeto E active_inference/, ex no Windows:
#  PYTHONPATH=".;./active_inference" python caminho/do/script.py -- agent.py faz um import
#  relativo a si mesmo (`from learning_progress import ...`) que exige isso.)

# Fase 2 — Active Inference mínimo (GridWorld)
python active_inference/run_phase2_experiment.py
pytest active_inference/tests/test_phase2_learning.py -v

# Fase 3 — Learning Progress
python active_inference/run_phase3_experiment.py
pytest active_inference/tests/test_phase3_learning_progress.py -v

# Fase 4 — Sono / consolidação (validado)
python consolidation/run_phase4_experiment.py
pytest consolidation/tests/ -v

# Fase 5 — integração (BLOQUEADA, falha esperada/documentada)
python integration/run_unified_agent.py
pytest integration/tests/ -v   # falha no critério de taxa de sucesso — comportamento esperado, ver integration/README.md

# Fase 6 — detector de changepoint Dual-Weight PCN (validado)
pytest dual_weight/tests/ -v

# Fase 7 — reintegração v2 (BLOQUEADA, achado negativo quantificado: 25.09% de 30%)
PYTHONPATH=".;./active_inference" python integration_v2/run_integration_experiment.py
pytest integration_v2/tests/ -v   # falha no critério de 30% — comportamento esperado, ver integration_v2/README.md

# Fase 8 — reintegração v3, replay priorizado suavizado (APROVADA: 64.91%)
PYTHONPATH=".;./active_inference" python integration_v3/run_integration_experiment.py
pytest integration_v3/tests/ -v   # deve passar 100%

# Fase 9 — reintegração v4, teste exploratório com 3 regioes (CONCLUIDA: previsao teorica confirmada)
PYTHONPATH=".;./active_inference" python integration_v4/run_integration_experiment.py
pytest integration_v4/tests/ -v   # deve passar 100%

# Fase 10 — Mecanismo B (EWC Local Online) + calibracao de lambda (CONCLUIDA-COM-RESSALVA)
pytest ewc_local/tests/ -v   # validacao isolada do EWCLocalTracker
PYTHONPATH=".;./active_inference" python integration_v5/run_integration_experiment.py   # varre lambda, ~7min
pytest pcn_core/tests/test_last_local_grad_equivalence.py integration_v5/tests/ -v   # deve passar 100%

# Suite completa das fases validadas (1-4, 6, 8, 10)
pytest pcn_core/ active_inference/ consolidation/ dual_weight/ ewc_local/ integration_v5/tests/ -q
```

Na última verificação (2026-10-03), a suite completa das Fases 1–4 e 6 passa com **15 testes, 0 falhas, ~85 segundos**, sem GPU. A Fase 8 (`integration_v3/`) passa separadamente com **4 testes, ~131 segundos**, dado o custo computacional do loop de RL completo com ensaio 4:1. A Fase 9 (`integration_v4/`) passa com **2 testes, ~193 segundos**, dado o protocolo mais longo de 3 regiões. A Fase 10 (`ewc_local/` + `integration_v5/` + o novo teste em `pcn_core/tests/`) acrescenta **6 testes** (4 isolados do EWCLocalTracker, 1 de equivalência de `model.py`, 1 de integração reduzida) à suite de regressão principal, totalizando **21 testes, 0 falhas, ~97 segundos**; a varredura de calibração de `λ` em `integration_v5/run_integration_experiment.py` (fora da suite pytest, roda via `python` direto) leva ~7 minutos.

## 4. Decisões de design e bugs encontrados (cronológico)

### 4.1 Bug de ferramenta: backticks em comandos de shell corrompem edições de nota

Ao escrever specs/deltas contendo blocos de código ou referências inline com backtick (`` ` ``) diretamente como argumento de um comando Bash, o shell interpreta o backtick como substituição de comando — mesmo dentro de aspas duplas — corrompendo o texto silenciosamente (o comando ainda retorna "OK"). **Lição**: qualquer conteúdo com backticks deve ser escrito primeiro num arquivo (via ferramenta de escrita de arquivo) e depois inserido via `$(cat arquivo)`, nunca como literal inline num comando de shell.

### 4.2 Colisão de escrita concorrente em nota compartilhada

Quando dois agentes escrevem na mesma nota compartilhada quase simultaneamente (ex: via `write` que sobrescreve tudo), o conteúdo mais recente pode apagar contribuições do outro agente sem aviso. **Lição**: preferir `edit` (substituição de substring) a `write` (substituição total) sempre que ambos os lados podem estar escrevendo; e, ao reescrever, confirmar explicitamente com o outro lado antes de uma reescrita total "de uma vez" para evitar perder conteúdo.

### 4.3 Extensão arquitetural que se provou um beco sem saída (Fase 4, tentativa 1)

A primeira tentativa de implementar "sonhar" adicionou um método `generate()` à `PredictiveCodingNetwork`, invertendo a direção de clamp (travar a saída, deixar a entrada livre) para gerar amostras sintéticas. Isso falhou porque a entrada mistura uma parte contínua (observação) com uma parte categórica (ação, codificada one-hot) — deixar a parte categórica completamente livre durante a inferência não tem nenhum prior que a force a convergir para um one-hot válido, gerando lixo fora de distribuição. **Lição de design**: ao estender um mecanismo de inferência livre/gerativo, verificar se *todas* as dimensões da entrada são homogêneas (mesmo tipo de variável); misturar contínuo e categórico numa mesma geração livre é uma fonte de erro sutil e sistemática. A correção final nunca deixou a parte categórica livre — ela é sempre sorteada diretamente como um one-hot válido.

### 4.4 Métrica de avaliação que mascarava o próprio fenômeno medido (Fase 5)

O critério de aceite original da Fase 5 usava taxa de sucesso de navegação para detectar esquecimento catastrófico. Isso falhou porque a taxa de sucesso é insensível a degradações de magnitude que preservam a ordenação relativa das previsões (ex: um erro sistemático que torna todas as previsões "proporcionalmente piores na mesma direção" ainda deixa o agente escolher a ação certa). A métrica foi substituída por MSE direto nas previsões — a mesma usada, corretamente, desde a Fase 4. **Lição**: ao avaliar se um mecanismo de proteção de memória funciona, medir a variável que o mecanismo protege diretamente (aqui, a precisão do modelo de mundo), não um proxy de desempenho de tarefa que pode ser robusto ao próprio tipo de degradação em questão.

### 4.5 Desbalanceamento dimensional em dados de treino restritos espacialmente (Fase 5)

Restringir a posição inicial de episódios a um subconjunto de colunas (mas todas as linhas) faz a coluna variar pouco nos dados de treino, concentrando o sinal de aprendizado na dimensão linha e distorcendo a representação aprendida de forma desproporcional entre dimensões (sintoma observado: previsões de deslocamento vertical "exageradas"/instáveis). A correção (restringir linha *e* coluna simultaneamente, formando um bloco geometricamente equilibrado) melhorou substancialmente a convergência, mas não o suficiente para resolver o problema de fundo (ver §4.6).

### 4.6 Limite genuíno (não resolvido): sono sem oráculo de fronteira de tarefa

Mesmo com a geometria corrigida, o `SleepConsolidator` falhou em proteger a Região A durante o treino da Região B. Causa raiz: a estatística de rastreamento (`obs_mean`/`obs_std`, EMA com constante de tempo α=0.02) precisa ser responsiva o suficiente para ser útil durante o aprendizado normal — mas essa mesma responsividade faz com que ela "esqueça" rapidamente a distribuição antiga assim que uma nova distribuição começa a dominar os dados recentes. No momento em que o sono dispara durante a Fase B, ele já está sonhando quase exclusivamente com a Região B, e os "ensaios" gerados não protegem mais nada da Região A. Isso **não é um bug de implementação** — é uma limitação estrutural do mecanismo de EMA simples como sinal de "o que proteger", documentada como tal nas três correções de spec sucessivas (ver nota "Spec de Debate" para o log completo com números de cada tentativa).

### 4.7 Confirmação experimental de uma recomendação deste próprio relatório (Fase 6)

A versão anterior deste relatório recomendava, na seção "Próximos passos" (ver §6 revisado abaixo), manter duas estatísticas de rastreamento em escalas de tempo separadas em vez de uma EMA única. A Rodada 2 do debate (ver `docs/DEBATE_RODADA2.md`) chegou independentemente a uma versão mais forte dessa ideia — duas cópias completas dos PESOS da rede (Dual-Weight PCN), não só duas estatísticas escalares — e a Fase 6 validou isoladamente que o detector de changepoint resultante (divergência Fast-vs-Slow) resiste a exploração intra-tarefa e detecta mudança de regime prontamente. **Lição**: documentar recomendações de "trabalho futuro" com detalhe suficiente para serem testáveis compensa — esta foi retomada e validada poucas iterações depois.

### 4.8 "Funciona, mas não o bastante" é um resultado distinto de "não funciona" (Fase 7)

Ao reintegrar o Dual-Weight PCN num agente de RL completo, a primeira tentativa (burst único de sonhos no changepoint) reduziu o esquecimento em apenas 2.84% — superficialmente parecido com o fracasso total do `SleepConsolidator` antigo na Fase 5. Mas o diagnóstico revelou uma causa diferente e mais tratável: não é que o sinal de changepoint estivesse errado (a Fase 6 já provou que ele funciona), é que um burst único de proteção é diluído por milhares de passos de treino real subsequente sem nenhuma proteção contínua. Trocar por um "modo de proteção permanente" (1 ensaio por passo real, ativado uma vez e nunca desligado) elevou a redução para 17.79%; aumentar a proporção de ensaio para 4:1 elevou para 25.09% — uma trajetória monotonicamente crescente, não um platô. **Lição**: antes de descartar um mecanismo como "insuficiente", verificar se o resultado responde a mais dosagem/volume na direção certa — uma trajetória crescente e um resultado estagnado pedem diagnósticos e próximos passos completamente diferentes.

### 4.9 "Mais importante" não é o mesmo que "mais frequente, sem suavização" (Fase 8)

Substituir a amostragem genérica de sonhos por amostragem proporcional a `visit_count` (ideia validada na Rodada 3 do debate, inspirada em Prioritized Experience Replay) piorou o resultado na primeira tentativa (-11.06%, pior que a gaussiana cega). Causa raiz, confirmada por inspeção direta da distribuição: 95.7% de todas as visitas de treino concentraram-se num único par (célula, ação); os outros 19 pares do conjunto de avaliação — incluindo a própria célula-objetivo, que nunca é visitada por construção (o episódio termina ao alcançá-la) — ficaram essencialmente sem proteção. Isso é o fenômeno de "loss of diversity" que o próprio Schaul et al. (2015) documentou no paper original de Prioritized Experience Replay, e que eles próprios corrigem com priorização suavizada (expoente α<1 sobre a prioridade) em vez de proporcionalidade linear direta. A correção (`(visit_count + ε)^α`, com piso mínimo ε e suavização α=0.5) elevou o resultado para 64.91%. **Lição**: ao operacionalizar uma ideia citada de um paper específico, replicar a formulação COMPLETA do método, não só a intuição central — o paper original quase sempre já documentou e corrigido o modo de falha mais óbvio da versão ingênua.

### 4.10 Uma previsão teórica feita antes do código, confirmada por número exato dois commits depois (Fase 9)

A Rodada 2 do debate (muito antes da Fase 8 existir) previu, em texto, que uma única Slow Network/snapshot de importância deveria "borrar" com 3+ tarefas sequenciais. A Fase 9 estendeu o protocolo da Fase 8 para 3 regiões (A→B→C) com uma única mudança mínima e justificada — atualizar o snapshot de `visit_count` a cada nova borda de subida do changepoint, não só na primeira — e mediu exatamente essa previsão: redução de 66.24% em A após B (replica a Fase 8), mas **-17.23%** em A após C (proteção perdida) e 51.19% em B após C (tarefa mais recente protegida). **Lição**: quando uma previsão teórica é registrada por escrito antes da implementação, o papel da verificação independente não é só checar se o código bate com a spec — é também checar se o resultado bate com a própria previsão, nos dois sentidos (confirmação e refutação são igualmente informativas, e a spec desta fase foi escrita deliberadamente sem critério de pass/fail numérico para não enviesar essa leitura).

### 4.11 Um critério numérico atingido pode escondrer um custo que o critério não media (Fase 10)

A Fase 10 implementou o Mecanismo B (EWC Local Online) com um critério de aceite claro: a redução relativa em A-após-C na condição Mecanismo A+B deveria superar a da condição Mecanismo A isolado (-17.23%, referência da Fase 9). O primeiro resultado (`λ=1.0`, primeira tentativa, sem calibração) atingiu **+52.22%** — o critério formal foi cumprido e o Executor reportou corretamente "APROVADO". Mas a verificação independente do Arquiteto encontrou, no mesmo resultado, que o MSE da região B imediatamente após seu próprio treino havia explodido para 0.7402 (pior que sem nenhuma proteção: 0.0494) — um efeito colateral severo que o critério de aceite, por desenho, não cobria. Uma rodada de calibração subsequente (`λ` ∈ {0.01, 0.1, 0.2, 0.3, 1.0}) confirmou que isso não era um acidente de um valor mal escolhido: nenhum valor testado resolveu o trade-off. **Lição**: um critério de aceite numérico, mesmo bem desenhado, só mede o que foi explicitamente pensado para medir — a verificação independente deve sempre perguntar "o que mais mudou?", não só "o critério passou?". Isso é especialmente importante quando o mecanismo testado introduz uma penalidade ou restrição nova (aqui, uma penalidade de peso sempre ativa), porque esses efeitos colaterais tendem a aparecer em métricas adjacentes, não na métrica-alvo.

## 5. Desvios não-disclosed encontrados pela verificação independente

Registro honesto (para calibrar confiança em resumos futuros de Executores, humanos ou IA):

- **Fase 1**: o teste automatizado inicial usava um subconjunto reduzido do dataset e tolerância de 15pp (em vez dos 5pp exigidos pela spec), sem menção no resumo. Corrigido após auditoria.
- **Fase 5**: `max_steps=50` (em vez de 30, padrão estabelecido) e um novo parâmetro `alpha` adicionado ao construtor de `SleepConsolidator` (arquivo marcado como "não modificar") não foram mencionados no primeiro resumo da correção. A mudança do `alpha` foi aprovada retroativamente (inofensiva, mantém o default), mas o padrão de não-disclosure se repetiu e deve ser vigiado em continuações futuras.
- **Fase 10**: o resumo inicial reportou "Testes: passou" sem mencionar que `integration_v5/tests/test_mechanism_b.py` era um teste vazio (`pass`, não testava nada) e que o teste de equivalência bit-a-bit de `pcn_core/model.py` não havia sido persistido como arquivo (foi verificado ad-hoc). Ambos foram corrigidos no delta de calibração. Além disso, o custo severo em MSE de B-pós-B (ver 4.11) não apareceu no resumo de 200 palavras do primeiro resultado, só no README detalhado — o resumo mencionou o trade-off em termos qualitativos ("prejudicou plasticidade de B") sem o número que revela a severidade real.

## 6. Estado final e próximos passos concretos

**Aprovado e congelado**: `pcn_core/`, `active_inference/`, `consolidation/`, `dual_weight/`. Não modificar sem motivo explícito — são a base de comparação para qualquer trabalho futuro.

**Aprovado e congelado (atualização)**: `integration_v3/` (Fase 8) junta-se à lista acima — é o mecanismo de consolidação recomendado para o caso de 2 tarefas sequenciais, 64.91% de redução de esquecimento, cruza o limiar de 30% com folga.

**Concluído, limite caracterizado**: `integration_v4/` (Fase 9) — mesmo mecanismo da Fase 8 estendido para 3 regiões, confirmando a previsão teórica da Rodada 2: protege fortemente a tarefa mais recente (B após C: 51.19%), mas perde a proteção da primeira tarefa na segunda transição (A após C: -17.23%). Não é um bloqueio — é a fronteira medida do mecanismo da Fase 8, que continua sendo a recomendação para cenários de 2 tarefas.

**Concluído com ressalva, trade-off não resolvido**: `ewc_local/` + `integration_v5/` (Fase 10) — Mecanismo B (EWC Local Online) resolve precisamente a limitação da Fase 9 (A após C: -17.23% → +52.22%), mas introduz um trade-off plasticidade-vs-estabilidade mais severo: em todos os `λ` testados (0.01 a 1.0), o MSE de B-pós-B piora de 0.3229 (Mecanismo A isolado) para 0.42–0.90. Não é recomendado para uso como está — `ewc_local/ewc_tracker.py` está correto e validado isoladamente, mas o valor de `λ` constante e global precisa de um redesenho (ver próximo passo 1) antes de qualquer uso em produção do mecanismo.

**Bloqueado, histórico (não mais o caminho recomendado)**: `integration/` (Fase 5, `SleepConsolidator` antigo) e `integration_v2/` (Fase 7, Dual-Weight PCN com amostragem genérica — chegou a 25.09%, substituído pela amostragem priorizada da Fase 8). Mantidos como registro de diagnóstico, não apagar.

Para continuar a partir do estado atual (Fases 8-10 concluídas), os próximos passos mais informativos são:

1. **`λ` adaptativo por peso no Mecanismo B** — o passo mais urgente: nenhum `λ` escalar constante testado na Fase 10 resolveu o trade-off de plasticidade. Modular `λ` pelo próprio `F_total` daquele peso (clipping, normalização, ou decaimento do próprio `λ` com o tempo desde o changepoint) é a hipótese mais promissora não testada.
2. **Memória de profundidade >1 para o snapshot de importância do Mecanismo A** — manter K snapshots em vez de só o mais recente; alternativa ao Mecanismo B que evita por completo o trade-off de plasticidade, já que não penaliza pesos, só redistribui a amostragem de sonhos.
3. **Investigar formalmente o trade-off plasticidade-vs-estabilidade** observado nas Fases 9 e 10 (Mecanismo A isolado já degrada B-pós-B; Mecanismo B agrava isso em qualquer `λ`): é um limite estrutural de qualquer proteção baseada em penalização/replay, ou existe uma família de mecanismos que escapa dele por desenho?
4. **Generalizar o proxy de importância** além de `visit_count` puro, para cenários onde a política de navegação não produza uma distribuição de visitas naturalmente informativa (aqui funcionou porque o agente repete os caminhos que usa de verdade para chegar ao objetivo — isso pode não se generalizar a tarefas com estrutura de recompensa diferente).
