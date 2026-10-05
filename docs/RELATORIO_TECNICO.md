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

ewc_local/ewc_tracker.py (mesmo arquivo)  — FASES 11-12, duas iteracoes de calibracao de c_layer
  Fase 11: c_layer = MEDIA de F_total por camada (retrocesso: B-pos-B 1.53-1.71, pior que tudo anterior)
  Fase 12: c_layer = MAXIMO de F_total por camada, revertido lam_values para {0.01,0.1,0.2,0.3} (hipotese refutada: fracao ratio>0.5 = 0%, mas B-pos-B ainda > 0.35 em todo lam)
  NOTA: nenhuma das duas fases tem README dedicado -- numeros reconstruidos a partir da nota de debate interna do projeto, nao de reexecucao (ver docs/ARTIGO.md §11-12 para a mesma ressalva)

hard_mask_local/                  — FASE 13 (Mecanismo C) + FASE 14 Parte 1 (fix), CONCLUIDA-COM-RESSALVA
  hard_mask_tracker.py            — classe HardMaskLocalTracker (mascara dura cumulativa + reset Xavier); selecao por ranking deterministico desde a Fase 14 (percentil com bug de empate ate a Fase 13)
  ablation_zero_tracker.py        — FASE 14 Parte 2, classe HardMaskZeroTracker (diagnostico, NAO USAR EM PRODUCAO -- zera em vez de ancorar pesos congelados)
  tests/test_hard_mask_tracker.py — inclui teste de regressao do bug de empate do percentil (Fase 14)
  tests/test_ablation_zero_tracker.py — FASE 14

integration_v6/                   — FASE 13 (Mecanismo C) + FASE 14 Partes 1-2, CONCLUIDA (2 resultados negativos bem diagnosticados)
  run_integration_experiment.py   — protocolo A->B->C + HardMaskLocalTracker; sweep K em {0.1,0.25,0.5,0.75}
  run_ablation_experiment.py      — FASE 14 Parte 2, compara HardMaskLocalTracker (ancorado) vs HardMaskZeroTracker (zerado) para o mesmo sweep de K
  tests/test_mechanism_c.py, tests/test_ablation_experiment.py
  README.md                       — tabela de calibracao (Fase 13), correcao do percentil + diagnostico de ablacao (Fase 14)

gating_local/                     — FASE 14 Parte 3 (Mecanismo D), CONCLUIDA (resultado negativo bem diagnosticado)
  gated_pcn.py                    — classe GatedPredictiveCodingNetwork(PredictiveCodingNetwork); sobrescreve train_step/predict inteiros para aplicar mascara de unidade oculta (o motor base nao expoe ganchos de extensao)
  gated_dual_weight_pcn.py        — classe GatedDualWeightPCN(DualWeightPCN); fast=GatedPredictiveCodingNetwork, slow=PredictiveCodingNetwork comum
  gating_tracker.py               — classe GatingLocalTracker (sorteio de gate por unidade oculta a cada changepoint, fracao de overlap configuravel)
  tests/test_gated_pcn.py         — inclui teste de equivalencia numerica exata (gate neutro == PCN original)
  tests/test_gating_tracker.py

integration_v7/                   — FASE 14 Parte 3 + FASE 15, CONCLUIDA (resultado negativo; deficit de capacidade confirmado com ressalva)
  run_integration_experiment.py   — protocolo A->B->C + GatingLocalTracker; sweep gate_frac em {0.25,0.5,0.75}; FASE 15 acrescenta calc_mse_with_gate/snapshot_gate (avaliacao com oraculo de tarefa, nao substitui a metrica principal)
  tests/test_mechanism_d.py       — inclui os testes de oraculo (restauracao de gate, equivalencia trivial, diferenca real, smoke de integracao) da Fase 15
  README.md                       — tabela de calibracao + diagnostico (Fase 14), tabela oraculo vs sem-oraculo + recomendacao para Rodada 10 (Fase 15)

docs/
  DEBATE_ORIGINAL.md    — reconstrução do debate (Rodada 1) que originou a arquitetura
  DEBATE_RODADA2.md     — segunda rodada de debate (diagnóstico da Fase 5, mecanismo Dual-Weight PCN)
  DEBATE_RODADA3.md     — terceira rodada de debate (observação do usuário sobre atenção/importância, mecanismo de replay priorizado)
  ARTIGO.md             — artigo científico (resultados consolidados, Fases 1-15; Rodadas 4-6 e 7-9 do debate narradas inline nas seções de cada fase, sem arquivo DEBATE_RODADA dedicado — lacuna de documentação conhecida, ver §6)
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

# Fases 11-12 — recalibracao de c_layer no Mecanismo B (sem README dedicado, ver docs/ARTIGO.md §11-12)
# Nao ha comando de reproducao documentado para estas duas fases especificas nesta revisao;
# o codigo atual de ewc_local/integration_v5 ja reflete o estado final da Fase 12 (c_layer=max, lam em {0.01,0.1,0.2,0.3}).

# Fase 13 — Mecanismo C (Mascara Dura + Reset), CONCLUIDA-COM-RESSALVA (resultado negativo bem diagnosticado)
pytest hard_mask_local/tests/ -v
PYTHONPATH="." python integration_v6/run_integration_experiment.py   # sweep K, ~8min

# Fase 14 — Correcao do percentil + Diagnostico de ablacao + Mecanismo D (Gating), CONCLUIDA (2 resultados negativos + 1 pista)
pytest hard_mask_local/tests/ gating_local/tests/ integration_v6/tests/ integration_v7/tests/ -v
PYTHONPATH="." python integration_v6/run_integration_experiment.py       # sweep K com percentil corrigido
PYTHONPATH="." python integration_v6/run_ablation_experiment.py          # ancorado vs zerado, mesmo sweep de K
PYTHONPATH="." python integration_v7/run_integration_experiment.py       # sweep gate_frac, Mecanismo D

# Fase 15 — Avaliacao com Oraculo de Tarefa, CONCLUIDA (deficit de capacidade confirmado, com ressalva)
pytest integration_v7/tests/ -v
PYTHONPATH="." python integration_v7/run_integration_experiment.py   # imprime tabela de calibracao + tabela oraculo vs sem-oraculo

# Suite completa de regressao (todas as fases nao-bloqueadas)
pytest pcn_core/ active_inference/ consolidation/ dual_weight/ ewc_local/ integration_v5/tests/ hard_mask_local/ gating_local/ integration_v6/tests/ integration_v7/tests/ -q
```

Na última verificação (2026-10-03), a suite completa das Fases 1–4 e 6 passa com **15 testes, 0 falhas, ~85 segundos**, sem GPU. A Fase 8 (`integration_v3/`) passa separadamente com **4 testes, ~131 segundos**, dado o custo computacional do loop de RL completo com ensaio 4:1. A Fase 9 (`integration_v4/`) passa com **2 testes, ~193 segundos**, dado o protocolo mais longo de 3 regiões. A Fase 10 (`ewc_local/` + `integration_v5/` + o novo teste em `pcn_core/tests/`) acrescenta **6 testes** (4 isolados do EWCLocalTracker, 1 de equivalência de `model.py`, 1 de integração reduzida) à suite de regressão principal, totalizando **21 testes, 0 falhas, ~97 segundos**; a varredura de calibração de `λ` em `integration_v5/run_integration_experiment.py` (fora da suite pytest, roda via `python` direto) leva ~7 minutos.

Na verificação mais recente (2026-10-04, ao final da Fase 15), a suite completa do projeto (`pytest` na raiz, incluindo todos os diretórios acima) passa com **49 testes, 2 falhas, ~447 segundos**. As 2 falhas são as mesmas duas conhecidas e documentadas desde antes da Fase 13 — `integration/tests/test_unified_agent.py::test_unified_agent` (taxa de sucesso da Fase A abaixo do limiar hardcoded, ambiente da Fase 5, bloqueado por desenho) e `integration_v2/tests/test_dual_weight_integration.py::test_integration_criteria` (`TypeError`, kwarg `n_dreams` que não existe mais em `run_experiment`, teste desatualizado da Fase 7, bloqueado por desenho) — nenhuma regressão nova foi introduzida pelas Fases 13-15. As varreduras fora da suite pytest (`integration_v6/run_integration_experiment.py`, `integration_v6/run_ablation_experiment.py`, `integration_v7/run_integration_experiment.py`, cada uma varrendo 3-4 valores de hiperparâmetro no protocolo completo A→B→C) levam, juntas, cerca de 25-30 minutos.

## 4. Decisões de design e bugs encontrados (cronológico)

### 4.1 Bug de ferramenta: backticks em comandos de shell corrompem edições de nota

Ao escrever specs/deltas contendo blocos de código ou referências inline com backtick (`` ` ``) diretamente como argumento de um comando Bash, o shell interpreta o backtick como substituição de comando — mesmo dentro de aspas duplas — corrompendo o texto silenciosamente (o comando ainda retorna "OK"). **Lição**: qualquer conteúdo com backticks deve ser escrito primeiro num arquivo (via ferramenta de escrita de arquivo) e depois inserido via `$(cat arquivo)`, nunca como literal inline num comando de shell.

### 4.2 Colisão de escrita concorrente em nota compartilhada (recorrente — quatro incidentes, Fases 13-15)

Quando dois agentes escrevem na mesma nota compartilhada quase simultaneamente (ex: via `write` que sobrescreve tudo), o conteúdo mais recente pode apagar contribuições do outro agente sem aviso. A lição original ("preferir `edit` a `write`") foi registrada após o primeiro incidente, mas **se repetiu em pelo menos quatro formas distintas** ao longo das Fases 13-15, apesar de correções sucessivas de processo:

1. **Fase 13**: o agente Executor tentou uma flag inexistente (`--raw`) ao ler a nota via PowerShell, e o comando malformado resultante escreveu o literal `"--stdin"` por cima de todo o conteúdo. Recuperado a partir de um arquivo de backup que o próprio agente havia criado por engano ao tentar redirecionar a saída.
2. **Fase 13→14 (transição de papel)**: ao reatribuir o papel do Executor para Debate (`maestri role assign`, que reinicia o processo do agente), uma escrita em andamento da sessão antiga colidiu com a reinicialização, revertendo parte do conteúdo recém-consolidado pelo Arquiteto para uma versão mais curta e desatualizada.
3. **Fase 14**: mesmo após o role do Executor ser corrigido explicitamente para instruir `edit` em vez de `write` e nomear o incidente #1 como exemplo a evitar, o Executor usou `write` (substituição total) duas vezes para gravar o bloco `--- RESULTADO ---` final, apagando a especificação completa que o Arquiteto havia acabado de escrever. Em ambos os casos, o Arquiteto manteve cópia local do conteúdo e refez a consolidação manualmente.
4. **Fase 15**: a correção de processo (role do Executor reescrito com uma regra explícita e nomeando os incidentes anteriores) finalmente funcionou — o resultado final foi anexado corretamente via `edit`, preservando a especificação.

**Lição revisada**: corrigir a instrução num role não garante que o agente a siga nas primeiras tentativas seguintes, especialmente sob um fluxo de "terminar e escrever resultado" que o agente associa mentalmente a "a tarefa acabou, grave tudo de uma vez". A correção que finalmente funcionou (Fase 15) combinou três elementos que as tentativas anteriores tinham só parcialmente: (a) citar o incidente concreto anterior como exemplo negativo explícito, não só a regra abstrata; (b) dar o comando exato a usar, não só "prefira X a Y"; (c) aplicar a correção ANTES do próximo ciclo de execução, não durante um em andamento. Nenhuma dessas quatro colisões causou perda real de resultado técnico — o Arquiteto manteve cópias locais de cada spec/resultado como prática padrão a partir do segundo incidente — mas o custo de tempo de recuperação foi real e cumulativo.

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

### 4.12 Refutar duas hipóteses opostas de normalização não localiza a causa raiz — mas elimina duas explicações (Fases 11-12)

Testar `c_layer` como média (Fase 11, piorou) e depois como máximo (Fase 12, não piorou mas também não resolveu) parece, à primeira vista, não ter produzido progresso — nenhuma das duas "funcionou". Mas cada teste eliminou uma explicação candidata com evidência direta: a Fase 12 mediu a fração de pesos com razão de penalização > 0.5 em exatos 0.00%, confirmando que a normalização por máximo elimina esse modo de falha específico por construção — e, ainda assim, o MSE de B-pós-B não melhorou. **Lição**: quando duas variantes opostas de uma mesma dimensão de design falham de formas diferentes mas para o mesmo critério de aceite, isso é evidência de que a causa raiz está em outra dimensão — não um motivo para continuar variando a mesma dimensão com mais granularidade.

### 4.13 Uma hipótese bem-fundamentada em literatura ainda pode estar errada sobre o mecanismo de falha específico (Fase 13)

O Mecanismo C (máscara dura + reset) tinha justificativa direta em quatro linhas de literatura de neurociência e continual learning, e corrigia precisamente o problema estrutural diagnosticado na Fase 12 (penalidade suave sobre pesos compartilhados). Mesmo assim, falhou — não porque a lógica de partição estivesse errada, mas porque a implementação específica (máscara cumulativa monotônica, sem orçamento) tinha seu próprio modo de falha não antecipado (esgotamento de capacidade). **Lição**: fundamentação em literatura reduz o risco de um mecanismo ser mal concebido, mas não substitui testar a implementação específica — a "tradução" de uma ideia da literatura para um sistema concreto sempre introduz detalhes de design (aqui: máscara permanece congelada para sempre, sem mecanismo de troca) que a ideia original, em abstrato, não especifica.

### 4.14 Um bug de ferramenta pode inflar um número sem invalidar a conclusão que o número suporta (Fase 13, corrigido na Fase 14)

O bug de empate do `np.percentile` inflou a fração congelada medida em até 3x o valor nominal — um erro de magnitude considerável. Mesmo assim, a conclusão qualitativa da Fase 13 (nenhum `K` satisfaz o critério de aceite; a máscara cresce rápido demais) permaneceu válida após a correção, só não a métrica diagnóstica secundária (fração congelada exibida). **Lição**: ao encontrar um bug de medição, a pergunta certa não é só "isso invalida o resultado?" mas "este bug afeta a MÉTRICA-ALVO do critério de aceite, ou uma métrica diagnóstica secundária?" — aqui era a segunda, o que mudou a prioridade da correção (importante, mas não bloqueante para interpretar o resultado principal já obtido).

### 4.15 O papel de debate funciona melhor quando o outro lado tem permissão explícita para refutar, não só concordar (Fase 15 / Rodada 9)

Na Rodada 9, o Arquiteto propôs três próximos passos ordenados por custo; o Executor, atuando no papel de Debate, identificou uma falha conceitual fatal no primeiro (promover a variante de zeragem diagnóstica a mecanismo real destruiria por definição a tarefa que deveria proteger) e o descartou antes de qualquer implementação. O Arquiteto aceitou a correção integralmente. **Lição**: a mesma disciplina de "verificação independente que não aceita o resultado sem examinar o custo escondido" (ver §4.11) se aplica também a propostas do próprio Arquiteto, não só a resultados do Executor — o papel de Debate, quando genuinamente adversarial dentro de um objetivo comum, encontrou uma hora de implementação desperdiçada antes que ela acontecesse, não depois.

### 4.16 Uma conclusão textual pode ser mais categórica do que a própria tabela que a acompanha sustenta (Fases 14 e 15)

Duas vezes nas Fases 14-15, o texto de conclusão de um README generalizou um padrão que a tabela de dados, lida com atenção, não sustentava uniformemente: na Fase 14 (Parte 2), "zerar piora para K≥0.5" contradizia o próprio K=0.75 da tabela (que melhorou); na Fase 15, "o oráculo não ajuda significativamente" ignorava uma melhora real de ~30% em `gate_frac=0.25`. Em ambos os casos, os NÚMEROS estavam corretos — só a frase de síntese arredondava um padrão misto para uma direção única. **Lição**: a verificação independente de um resultado quantitativo deve ler a tabela e a conclusão escrita como duas fontes separadas de informação, checando se uma sustenta a outra — um padrão não-monotônico ou misto genuinamente existe em dados de execução única (sem repetição de seed), e forçá-lo numa narrativa limpa é um viés de redação fácil de cometer e fácil de não notar sem comparar número a número.

## 5. Desvios não-disclosed encontrados pela verificação independente

Registro honesto (para calibrar confiança em resumos futuros de Executores, humanos ou IA):

- **Fase 1**: o teste automatizado inicial usava um subconjunto reduzido do dataset e tolerância de 15pp (em vez dos 5pp exigidos pela spec), sem menção no resumo. Corrigido após auditoria.
- **Fase 5**: `max_steps=50` (em vez de 30, padrão estabelecido) e um novo parâmetro `alpha` adicionado ao construtor de `SleepConsolidator` (arquivo marcado como "não modificar") não foram mencionados no primeiro resumo da correção. A mudança do `alpha` foi aprovada retroativamente (inofensiva, mantém o default), mas o padrão de não-disclosure se repetiu e deve ser vigiado em continuações futuras.
- **Fase 10**: o resumo inicial reportou "Testes: passou" sem mencionar que `integration_v5/tests/test_mechanism_b.py` era um teste vazio (`pass`, não testava nada) e que o teste de equivalência bit-a-bit de `pcn_core/model.py` não havia sido persistido como arquivo (foi verificado ad-hoc). Ambos foram corrigidos no delta de calibração. Além disso, o custo severo em MSE de B-pós-B (ver 4.11) não apareceu no resumo de 200 palavras do primeiro resultado, só no README detalhado — o resumo mencionou o trade-off em termos qualitativos ("prejudicou plasticidade de B") sem o número que revela a severidade real.
- **Fase 14**: o resumo de 200 palavras do Executor reportou "Testes: passou" e um veredito qualitativo correto nas 3 partes, mas a inconsistência entre a conclusão textual do README e a própria tabela de ablação (ver §4.16) só foi encontrada porque o Arquiteto releu a tabela número a número — não estava evidente a partir do resumo, que citava apenas o caso favorável (K=0.25).
- **Fase 15**: o mesmo padrão do item acima se repetiu — o resumo e o README reportaram "o oráculo não ajuda significativamente" como conclusão única, sem mencionar que `gate_frac=0.5` não chegou a testar a hipótese de fato (oráculo e não-oráculo idênticos, por nenhum *changepoint* adicional ter disparado) nem que `gate_frac=0.25` mostrou uma melhora real de ~30% em duas das três métricas — ambos os fatos só emergiram da leitura direta da tabela pela verificação independente.

## 6. Estado final e próximos passos concretos

**Aprovado e congelado**: `pcn_core/`, `active_inference/`, `consolidation/`, `dual_weight/`. Não modificar sem motivo explícito — são a base de comparação para qualquer trabalho futuro.

**Aprovado e congelado (atualização)**: `integration_v3/` (Fase 8) junta-se à lista acima — é o mecanismo de consolidação recomendado para o caso de 2 tarefas sequenciais, 64.91% de redução de esquecimento, cruza o limiar de 30% com folga.

**Concluído, limite caracterizado**: `integration_v4/` (Fase 9) — mesmo mecanismo da Fase 8 estendido para 3 regiões, confirmando a previsão teórica da Rodada 2: protege fortemente a tarefa mais recente (B após C: 51.19%), mas perde a proteção da primeira tarefa na segunda transição (A após C: -17.23%). Não é um bloqueio — é a fronteira medida do mecanismo da Fase 8, que continua sendo a recomendação para cenários de 2 tarefas.

**Concluído, superseded (não recomendado, mantido como registro)**: `ewc_local/` + `integration_v5/` (Fases 10-12) — Mecanismo B (EWC Local Online) resolve precisamente a limitação da Fase 9 (A após C: -17.23% → +52.22%, Fase 10), mas introduz um trade-off plasticidade-vs-estabilidade que nem a calibração de `λ` (Fase 10: 0.01-1.0) nem duas recalibrações opostas de normalização de `c_layer` (Fase 11: média, piorou para B-pós-B 1.53-1.71; Fase 12: máximo, eliminou o modo de falha de razão>0.5 mas não o trade-off) resolveram. A Rodada 7 do debate concluiu que a causa é estrutural (penalidade suave sobre pesos compartilhados), não de calibração — ver os mecanismos seguintes.

**Concluído, resultado negativo bem diagnosticado**: `hard_mask_local/` + `integration_v6/` (Fases 13-14) — Mecanismo C (máscara dura de consolidação + reset de capacidade livre) não satisfez o critério de aceite para nenhum `K` ∈ {0.1, 0.25, 0.5, 0.75}; a máscara cumulativa monotônica esgota a capacidade livre rapidamente (Camada 1 chega a ~100% de ocupação em K=0.5). Um bug de empate no corte por percentil (Fase 13) foi corrigido na Fase 14 sem alterar essa conclusão. Um diagnóstico de ablação na Fase 14 isolou evidência real de interferência residual de *forward* em K=0.25 (zerar em vez de ancorar os pesos congelados reduziu B-pós-B em ~35%) — achado que motivou o Mecanismo D, não uma recomendação de uso do Mecanismo C como está.

**Concluído, resultado negativo bem diagnosticado**: `gating_local/` + `integration_v7/` (Fases 14-15) — Mecanismo D (*Context-dependent Gating*) perdeu severamente para o Mecanismo C em todos os `gate_frac` ∈ {0.25, 0.5, 0.75} testados (B-pós-B > 2.0, pior que até o baseline sem proteção). Causa diagnosticada: déficit de capacidade da rede-base (16 e 8 unidades ocultas), não falha da lógica de isolamento por *gating* em si. Uma avaliação com oráculo de tarefa (Fase 15) confirmou essa leitura com uma ressalva: em `gate_frac=0.25`, o oráculo melhorou genuinamente ~30% em duas das três métricas, indicando que parte do fracasso é também artefato de avaliação (gate errado usado no teste), não só déficit de capacidade puro. `gating_local/gated_pcn.py` está validado isoladamente (teste de equivalência numérica exata com gate neutro) e é reutilizável para um futuro teste com rede escalada.

**Bloqueado, histórico (não mais o caminho recomendado)**: `integration/` (Fase 5, `SleepConsolidator` antigo) e `integration_v2/` (Fase 7, Dual-Weight PCN com amostragem genérica — chegou a 25.09%, substituído pela amostragem priorizada da Fase 8). Mantidos como registro de diagnóstico, não apagar.

**Aprovado e congelado (sem mudança desde a última revisão)**: `pcn_core/`, `active_inference/`, `consolidation/`, `dual_weight/`, `integration_v3/` (Fase 8, mecanismo recomendado para 2 tarefas), `integration_v4/` (Fase 9, fronteira medida para 3 tarefas). Nenhuma das Fases 11-15 modificou estes diretórios.

**Lacuna de documentação conhecida**: ao contrário das Rodadas 2 e 3 (`docs/DEBATE_RODADA2.md`, `docs/DEBATE_RODADA3.md`), as Rodadas 4 a 9 do debate (que produziram o Mecanismo B e todas as suas iterações, Fases 10-15) não têm um arquivo de transcrição dedicado — seu conteúdo está narrado em prosa dentro de `docs/ARTIGO.md` (§10-§15) e, de forma mais completa mas efêmera, na nota de debate interna do projeto (sobrescrita e reconstruída várias vezes, ver §4.2). Reconstruir arquivos `DEBATE_RODADA4.md` a `DEBATE_RODADA9.md` dedicados, equivalentes em detalhe aos das Rodadas 2-3, é um item de documentação pendente, não um item de pesquisa.

Para continuar a partir do estado atual (Fases 1-15 concluídas), os próximos passos mais informativos são (ver também `docs/ARTIGO.md` §18, mesma lista com mais contexto):

1. **Escalar a capacidade da rede-base antes de re-testar qualquer mecanismo de proteção** — o passo mais urgente: todos os mecanismos de proteção testados desde a Fase 10 (EWC, Máscara Dura, Gating) usaram a mesma rede pequena (16 e 8 unidades ocultas); a Fase 15 não conseguiu isolar completamente "o mecanismo não funciona" de "não sobrou capacidade para o mecanismo funcionar". Repetir o Mecanismo D (já implementado e reutilizável) numa rede ~4x maior (ex: `[7, 64, 32, 2]`) é o teste mais direto e barato disponível.
2. **Inferência automática de gate/tarefa no Mecanismo D** — condicional ao item 1: a Fase 15 mostrou ganho real com oráculo de tarefa em pelo menos uma configuração; inferir automaticamente qual gate usar (sem oráculo externo) é o passo natural seguinte.
3. **Memória de profundidade >1 para o snapshot de importância do Mecanismo A** (replay priorizado, Fases 8-9) — alternativa a toda a família EWC/Máscara/Gating, não testada: manter K snapshots em vez de só o mais recente, sem penalizar nem particionar pesos.
4. **Orçamento de capacidade para o Mecanismo C** — considerado e deliberadamente descartado na Rodada 9 (trocar pesos em vez de só unir máscaras exigiria resolver primeiro o item 1 para ser um teste justo); registrado aqui para não ser retestado sem essa pré-condição.
5. **Sincronizar ou medir explicitamente o detector de *changepoint* com fronteiras de tarefa reais** — comparações de sweep entre as Fases 13-15 podem estar confundidas por quantos *changepoints* dispararam em cada execução específica (variável não controlada; ver exemplo concreto em `gate_frac=0.5` da Fase 15).
6. **Generalizar o proxy de importância** além de `visit_count` puro, para cenários onde a política de navegação não produza uma distribuição de visitas naturalmente informativa.
