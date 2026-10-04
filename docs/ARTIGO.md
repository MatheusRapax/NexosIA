# Developmental PCN: Aprendizado Local, Motivação Intrínseca e Consolidação Sem Buffer — Prova de Conceito, Quatro Gerações do Mecanismo de Consolidação, e o Trade-off que Resta

**Tipo:** Relatório técnico-científico de prova de conceito (proof-of-concept), com dois achados negativos quantificados, uma terceira iteração que cruzou o limiar de aceite, uma quarta investigação que mediu a fronteira desse sucesso, e uma quinta que resolveu essa fronteira ao custo de um novo trade-off, ainda sem solução.
**Data:** Setembro–Outubro de 2026.
**Autores (processo):** Arquitetura concebida em debate estruturado entre dois agentes de IA (Claude Code e Antigravity, via Maestri); implementação conduzida sob fluxo Arquiteto/Executor, com verificação independente de cada entrega; mecanismo de consolidação revisado em quatro rodadas de debate sucessivas, a segunda, terceira e quarta iniciadas por observações ou decisões do usuário humano.

## Resumo

Modelos de linguagem de grande escala (LLMs) dependem de computação massiva porque separam memória de longo prazo (pesos, congelados após o treino) de memória de trabalho (contexto, descartada a cada sessão), forçando recomputação bruta via atenção a cada inferência. Este trabalho implementa e testa empiricamente uma arquitetura alternativa — a **Developmental PCN** — que unifica memória e cômputo num único substrato de pesos plásticos, aprendendo por regras estritamente locais (sem backpropagation global). A arquitetura tem quatro pilares: (1) um motor de Predictive Coding Network (PCN) matematicamente equivalente a backprop mas implementável com message passing local; (2) Active Inference como princípio unificador de percepção e ação; (3) Learning Progress como motivação intrínseca, substituindo erro de predição bruto; (4) consolidação via "sono", sem buffer de experiências reais. Validamos cada pilar isoladamente com sucesso (Fases 1–4), obtendo: paridade com um baseline de backprop em classificação (-0.56 pontos percentuais); navegação bem-sucedida num ambiente simulado usando o mesmo substrato para memória e ação (100% de sucesso); resistência empírica ao "Noisy TV problem" via Learning Progress; e redução de 66.57% no esquecimento catastrófico via sono generativo (num cenário com fronteira de tarefa conhecida). A primeira tentativa de integrar os quatro pilares num único agente contínuo e *task-free* (Fase 5) **falhou em quatro iterações sucessivas**, com o mecanismo de consolidação original produzindo proteção próxima de zero independentemente da calibração — um limite estrutural, não um problema de ajuste fino. Uma segunda rodada de debate diagnosticou a causa raiz ("generator forgetting") e propôs um mecanismo substituto, o **Dual-Weight PCN**, validado isoladamente (Fase 6). Sua reintegração (Fase 7) produziu uma trajetória monotonicamente crescente — 2.84% → 17.79% → 25.09% — mas ainda abaixo do limiar de 30%. Uma terceira rodada de debate, iniciada por uma observação do usuário sobre como o cérebro usa atenção e importância para não processar tudo igualmente, levou a uma segunda troca de mecanismo: substituir a amostragem genérica de sonhos por **replay priorizado** com base na importância real de cada transição (inspirado em Prioritized Experience Replay e Elastic Weight Consolidation). Uma primeira versão ingênua desse mecanismo piorou o resultado para -11.06% — um efeito de "perda de diversidade" já documentado na literatura de origem — mas, corrigida com priorização suavizada, produziu **64.91% de redução**, mais que o dobro do melhor resultado anterior e muito acima do limiar de aceite (Fase 8). Uma investigação seguinte (Fase 9) estendeu esse mecanismo vencedor para 3 regiões sequenciais (A→B→C) e mediu exatamente onde seu sucesso tem fronteira: a tarefa imediatamente anterior à transição mais recente continua fortemente protegida (51.19% de redução), mas a proteção da primeira tarefa não sobrevive à segunda transição (-17.23%) — confirmando, com números precisos, a previsão feita na segunda rodada de debate de que um único snapshot de importância só protege "uma tarefa por vez". Uma quarta rodada de debate projetou o **Mecanismo B** (EWC Local Online, adaptando Kirkpatrick et al. 2017 e Schwarz et al. 2018 ao proxy de Fisher local já presente na regra de aprendizado do PCN) como complemento, não substituto, ao replay priorizado. Testado (Fase 10), o Mecanismo B resolveu precisamente o problema medido na Fase 9 — a redução em A-após-C subiu de -17.23% para **+52.22%** — mas uma calibração subsequente revelou um custo que nenhum valor de `λ` testado (0.01 a 1.0) resolveu: a capacidade da rede de aprender a tarefa seguinte (B) degrada severamente sob a penalidade, de forma que a "vitória" em reter a tarefa mais antiga chega ao preço de prejudicar o aprendizado da mais nova. O trabalho documenta, portanto, cinco gerações sucessivas de investigação sobre o mesmo problema: a primeira confirmou um limite teórico já antecipado; a segunda avançou substancialmente sem cruzá-lo; a terceira, originada de uma intuição biológica do usuário humano e refinada por uma autocorreção rigorosa, o cruzou com folga para duas tarefas; a quarta mediu com precisão onde essa vitória termina; a quinta resolveu esse limite específico, mas revelou um trade-off estrutural de plasticidade-vs-estabilidade ainda sem solução.

## 1. Introdução e Motivação

A motivação original deste projeto nasce de uma observação simples: modelos de linguagem modernos são extraordinariamente ineficientes em comparação ao cérebro humano. Um LLM precisa de milhões de operações de ponto flutuante para prever o próximo token; o cérebro humano opera com ~20 W de potência e usa representações esparsas e plasticidade sináptica contínua. Parte dessa ineficiência não é acidental — é estrutural: LLMs separam **peso** (conhecimento de longo prazo, fixado no treino) de **contexto** (memória de trabalho, efêmera, descartada a cada nova sessão). Essa separação força o modelo a "reconstruir" seu raciocínio do zero a cada chamada, via atenção bruta sobre todo o contexto disponível.

A pergunta motivadora foi: é possível construir uma inteligência artificial que (a) não dependa de escala bruta de hardware do jeito que LLMs dependem, e (b) cresça genuinamente por experiência, com memória e raciocínio unificados no mesmo substrato — como no cérebro humano?

A arquitetura aqui testada emergiu de um debate estruturado entre dois agentes de IA, documentado in extenso em [DEBATE_ORIGINAL.md](DEBATE_ORIGINAL.md). O debate é parte integrante da metodologia deste trabalho: cada pilar da arquitetura nasceu de um ponto e foi imediatamente submetido a contraponto rigoroso antes de ser aceito, e as limitações conhecidas foram registradas **antes** de qualquer implementação.

## 2. Trabalhos Relacionados

| Linha de pesquisa | Referência | Contribuição para este trabalho |
|---|---|---|
| Memória como computação em test-time | Behrouz, Zhong & Mirrokni, "Titans: Learning to Memorize at Test Time", arXiv:2501.00663 (2024) | Demonstra que um módulo neural pode atualizar os próprios pesos durante a inferência; motivou o pilar de memória-como-cômputo, embora ainda dependente de backprop. |
| Complementary Learning Systems | McClelland et al. (1995); Kumaran, Hassabis & McClelland (2016) | Modelo biológico de consolidação hipocampo→córtex via replay offline; inspirou o mecanismo de "sono". |
| Neuromorphic computing / Spiking Neural Networks | IBM NorthPole (2023); revisões gerais de SNN | Demonstra viabilidade de hardware event-driven de baixo consumo; motivou o requisito de regras de atualização locais. |
| World Models / JEPA | LeCun, "Objective-Driven AI" (2024); V-JEPA 2 (Meta, 2025) | Aprendizado de representações preditivas latentes para planejamento; influenciou o design do agente embodied. |
| Active Inference / Free Energy Principle | Friston et al., "Uncertainty, epistemics and active inference" (2017); Friston, "Bayesian model reduction" arXiv:1805.07092 (2018) | Fundamenta a unificação percepção-ação via minimização de Energia Livre (Esperada); resolve formalmente o "Dark Room problem". |
| Predictive Coding como alternativa local a backprop | Whittington & Bogacz, "An approximation of the error backpropagation algorithm in a predictive coding network" (2017); Millidge, Song et al., "Predictive Coding Can Do Exact Backpropagation on CNNs and RNNs" (2021) | Prova que PCN com regras puramente locais aproxima (ou reproduz exatamente) o gradiente de backprop — o pilar central deste trabalho. |
| Motivação intrínseca e exploração | Oudeyer & Kaplan, "Intrinsic motivation, curiosity and learning" (Learning Progress Hypothesis); Burda et al., "Large-Scale Study of Curiosity-Driven Learning" (2018, Noisy TV problem) | Fundamenta o uso de taxa de redução de erro (em vez de erro bruto) como sinal epistêmico. |
| Generative replay sem buffer | van de Ven, Siegelmann & Tolias, "Brain-inspired replay for continual learning with artificial neural networks", *Nature Communications* 11:4069 (2020) | Precedente empírico revisado por pares para replay interno gerado pela própria rede, sem armazenamento de dados brutos. |

## 3. Arquitetura Proposta

A Developmental PCN é composta por quatro pilares, cada um implementado e testado numa fase distinta do projeto:

### Pilar 1 — Motor Computacional (PCN)
Rede com camadas de "nós de valor" (`x_0 ... x_L`), onde cada camada prediz a seguinte via `μ_l = tanh(W_l·x_{l-1} + b_l)`. O erro de predição `e_l = x_l - μ_l` define uma energia total `E = Σ 0.5·||e_l||²`. A **inferência** minimiza `E` ajustando iterativamente os nós de valor livres via gradiente local (`dx_l = -e_l + W_{l+1}ᵗ·e_{l+1}·f'(...)`), usando apenas informação de camadas vizinhas. O **aprendizado** atualiza os pesos via regra Hebbiana local (`ΔW_l = η·e_l⊗x_{l-1}`), também usando só informação local. Nenhuma chamada a autograd/backpropagation é usada em nenhum momento.

### Pilar 2 — Active Inference (percepção e ação unificadas)
Um "agente" usa a mesma PCN como modelo de mundo. Para selecionar uma ação, prediz o resultado de cada ação candidata (`world_model.predict`) e escolhe a que minimiza a distância prevista a um estado-objetivo (valor pragmático da Energia Livre Esperada). A memória do agente **é** a malha de pesos do world model — não há um módulo de memória separado (KV-cache, buffer de replay, etc.).

### Pilar 3 — Learning Progress (motivação intrínseca)
A exploração não é guiada por erro de predição bruto (vulnerável ao *Noisy TV problem*) nem por aleatoriedade explícita (epsilon-greedy), mas por um `LearningProgressTracker`: para cada par (região, ação), mantém-se duas médias móveis exponenciais do erro com constantes de tempo diferentes (rápida e lenta); a diferença `slow_ema - fast_ema` mede a *taxa de melhora* — alta quando o agente está genuinamente aprendendo algo novo, próxima de zero quando o sinal é ruído irredutível. A seleção de ação minimiza `EFE = pragmatic_weight·G - epistemic_weight·LP`, combinando objetivo e curiosidade num único valor escalar, sem nenhuma chamada a geradores de números aleatórios na política.

### Pilar 4 — Consolidação ("Sono")
Um `SleepConsolidator` mantém estatísticas agregadas de tamanho fixo (média e desvio das observações, EMA do erro crônico) — nunca uma lista crescente de experiências reais. Periodicamente, "sonha": amostra observações da distribuição aprendida, combina com ações válidas sorteadas, e congela a predição atual do modelo como alvo de "ensaio" (`dream_memory`, tamanho fixo). Esses sonhos são revividos (`rehearse`) durante o aprendizado de novas informações, protegendo o conhecimento antigo sem nunca armazenar uma transição real literal. Adicionalmente, pesos associados a erro cronicamente alto sofrem poda (weight decay proporcional), aproximando Bayesian Model Reduction sem o custo de comparação explícita de modelos.

## 4. Metodologia Experimental e Resultados por Fase

Cada fase foi especificada com critérios de aceite mensuráveis, implementada por um agente Executor, e **verificada de forma independente pelo Arquiteto** (re-execução dos experimentos e inspeção direta do código, não apenas leitura do resumo reportado).

### Fase 1 — Motor PCN vs. Backpropagation

**Tarefa:** classificação supervisionada no dataset `digits` do scikit-learn (8×8 pixels, 10 classes, 1797 amostras, split 80/20).
**Topologia:** `[64, 32, 16, 10]`, idêntica para PCN e baseline MLP com backprop (PyTorch, SGD).
**Protocolo:** 30 épocas, mesma seed.

| Modelo | Acurácia |
|---|---|
| PCN (local, numpy, sem autograd) | 96.67% |
| Baseline (backprop, PyTorch) | 97.22% |
| **Diferença** | **-0.56 pontos percentuais** |

Critério de aceite (diferença ≤ 5pp) cumprido com folga. Suite de testes: 2 testes, ~66s.

### Fase 2 — Active Inference Mínimo (Valor Pragmático)

**Tarefa:** navegação num `GridWorld` 5×5, objetivo fixo no canto (4,4), posição inicial uniforme, 5 ações discretas, 250 episódios.
**Mecanismo:** seleção de ação por minimização de distância prevista ao objetivo (usando a mesma PCN da Fase 1 como modelo de mundo), com exploração via epsilon-greedy decrescente (placeholder explícito, substituído na Fase 3).

| Agente | Janela inicial (50 ep.) | Janela final (50 ep.) |
|---|---|---|
| PCN-AIF | 68.0% | **100.0%** |
| Agente aleatório | 24.0% | 16.0% |

Evolução do PCN-AIF: +32pp. Vantagem final sobre aleatório: +84pp.

### Fase 3 — Learning Progress (Valor Epistêmico)

**Mudança:** epsilon-greedy substituído por `EFE = pragmatic_weight·G - epistemic_weight·LP`, 100% determinístico (zero chamadas a geradores aleatórios na política).

**Testes unitários do `LearningProgressTracker`** (isolados do loop de RL, provando as propriedades teóricas diretamente):
- Sequência de erro monotonicamente decrescente → LP > 0.01 (detecta progresso real).
- Sequência de erro ruidosa sem tendência (`1.0 + N(0, 0.05)`) → |LP| < 0.05 (resiste a ruído irredutível — prova direta da resistência ao Noisy TV problem).
- Par (região, ação) nunca visitado → LP = valor de bootstrap exato.

**Resultado no loop completo:**

| Métrica | Valor |
|---|---|
| Taxa de sucesso (janela final) | 94.0% |
| Vantagem sobre agente aleatório | +62pp (94% vs 32%) |
| Energia média do world model (início) | 0.02642 |
| Energia média do world model (fim) | 0.00423 |
| **Redução de erro** | **-84%** (requerido: ≥50%) |

Nota metodológica relevante: a taxa de sucesso já começa alta (94%) na primeira janela, porque a exploração guiada por Learning Progress é eficiente o suficiente para atingir quase-otimalidade rapidamente num ambiente pequeno. Por isso, o critério de aceite original ("melhora de 20pp entre janelas") foi corrigido para incluir a queda de energia do world model como evidência independente de aprendizado real — uma decisão metodológica que se revelaria crucial na Fase 5 (ver §6).

### Fase 4 — Consolidação via Sono (Bayesian Model Reduction Aproximado)

**Tarefa:** esquecimento catastrófico controlado. Um dataset sintético exaustivo de 125 transições (25 células × 5 ações) do GridWorld é dividido em Região A (colunas 0–1, 50 transições) e Região B (colunas 3–4, 50 transições). O mesmo world model é treinado sequencialmente em A, depois em B, com e sem intervenção de sono entre as duas fases.

**Primeira tentativa** (adição de um método `generate()` à PCN, invertendo a direção de clamp para "sonhar" livremente a entrada): **falhou**. A porção de ação do vetor de entrada (codificação one-hot) não tem nenhum prior que a restrinja a ser um one-hot válido quando deixada livre — convergia para vetores fora de distribuição, corrompendo os pesos. O Executor reportou corretamente o bloqueio em vez de mascarar o problema.

**Design corrigido**: a ação dos sonhos é sempre sorteada como um one-hot válido (nunca "gerada livremente"); o alvo do sonho é um snapshot congelado da predição atual do modelo, capturado uma única vez no limite entre tarefas, e revivido (`rehearse`) a cada época do treino subsequente — não apenas uma vez.

| Condição | Erro em Região A (MSE) |
|---|---|
| Após treinar em A | 0.02228 |
| Após treinar em B, **sem** sono | 0.23511 |
| Após treinar em B, **com** sono | 0.07859 |
| **Redução relativa de erro** | **66.57%** (requerido: ≥30%) |

13 testes automatizados (cumulativos de todas as fases) passam sem regressão, suite completa executando em ~100s.

## 5. Fase 5 — Tentativa de Integração: Achado Negativo

O objetivo da Fase 5 era combinar os quatro pilares num único agente contínuo, com sono disparado **periodicamente** (por contagem de episódios, sem conhecer de antemão uma fronteira de tarefa) em vez de no limite conhecido entre tarefas (como na Fase 4). Esse é um passo deliberado em direção ao problema de *task-free continual learning*, identificado como aberto desde a concepção teórica da arquitetura.

Foram necessárias quatro iterações, cada uma expondo uma camada mais profunda do problema:

**Tentativa 1.** Trocar apenas o objetivo do GridWorld (sem restringir a região de início) não produz esquecimento real — a física de movimento é uniforme em todo o grid, então o world model não "esquece" nada ao mudar o alvo.

**Tentativa 2.** Restringir o início a uma faixa estreita de colunas (para forçar concentração de experiência, replicando a condição que causou esquecimento na Fase 4) revelou um defeito de **métrica**: a taxa de sucesso de navegação permaneceu em 100% em ambas as condições (com e sem sono), mesmo com o erro (MSE) do world model crescendo substancialmente. A causa é que a *topologia relativa* das predições (qual direção é "mais perto do objetivo") sobrevive a degradações absolutas de magnitude — a tarefa de navegação, por desenho, é robusta a exatamente o tipo de erro que o sono deveria prevenir. **Taxa de sucesso provou ser uma métrica inadequada para detectar esse fenômeno**; a métrica correta é o MSE direto, como já usado na Fase 4.

**Tentativa 3.** Corrigida a métrica (MSE), a linha de base da Região A (antes de qualquer treino em B) não atingiu o mínimo de competência exigido (36–48%, contra um objetivo de ≥70%). Hipóteses de motivação intrínseca excessiva (`epistemic_weight`) foram testadas e descartadas (valores 0.0 e 0.2 produziram resultados idênticos). O diagnóstico convergiu para um efeito geométrico: restringir o início por uma única dimensão (coluna) faz essa dimensão variar muito pouco nos dados de treino, concentrando quase todo o sinal de aprendizado na dimensão ortogonal (linha) e distorcendo desproporcionalmente a representação aprendida pela rede densa ("warping espacial").

**Tentativa 4 (final).** Corrigida a geometria (restrição por linha *e* coluna simultaneamente, formando um bloco 2×2 balanceado, com objetivos nos cantos — a mesma geometria validada nas Fases 2–3), o MSE da Região A melhorou substancialmente (de ~0.89 para ~0.224), mas **ainda não o suficiente**: o desvio-padrão do erro residual (~0.47) é maior que o próprio tamanho do passo de movimento do grid (0.25) — o ruído absoluto afoga a informação direcional necessária para navegação, mesmo com a geometria corrigida. Adicionalmente, o mecanismo de sono, cuja estatística (EMA com constante de tempo rápida, α=0.02) foi desenhada para rastrear a distribuição *atual* de observações, esquece a Região A assim que a Região B começa a dominar os dados recentes — e passa a "sonhar" quase exclusivamente com a Região B, falhando em proteger o conhecimento antigo. A redução relativa de erro na condição com sono foi de **-0.02%** (nula), muito abaixo do limiar de 30%.

**Resultado final: bloqueado.** Conforme protocolo acordado, a quarta tentativa foi a última autorizada; o experimento foi documentado como achado negativo em vez de forçado a "passar" por meios artificiais.

## 6. Fase 6 — Validação Isolada de um Mecanismo Substituto (Dual-Weight PCN)

A Fase 5 revelou que a causa raiz do fracasso era o `SleepConsolidator` original rastrear uma única estatística gaussiana de observação, que por precisar ser responsiva o suficiente para ser útil no presente, inevitavelmente esquece o passado assim que a distribuição recente muda — uma instância do problema conhecido de "generator forgetting" em generative replay contínuo (Aljundi et al., "Task-Free Continual Learning", CVPR 2019). Uma segunda rodada de debate estruturado entre os dois agentes (ver [DEBATE_RODADA2.md](DEBATE_RODADA2.md)) considerou e descartou duas propostas (protótipos múltiplos ao estilo CoPE; sono disparado por Bayesian Online Changepoint Detection) antes de convergir no mecanismo **Dual-Weight PCN**: duas cópias dos pesos da PCN — uma *Fast Network* que aprende online normalmente, e uma *Slow Network* nunca treinada diretamente em dados, só atualizada via Polyak averaging (EMA) dos pesos da Fast. A divergência de energia entre as duas redes sobre a mesma transição (`delta`, suavizado por EMA escalar) serve de gatilho de changepoint — por ser relativo (dois modelos competindo no mesmo dado), resiste à exploração intra-tarefa do Learning Progress, ao contrário de um detector de erro absoluto.

**Validação isolada** (sem RL, sem GridWorld, dados sintéticos — mesma disciplina que validou o `LearningProgressTracker` na Fase 3):

| Teste | Critério | Resultado |
|---|---|---|
| Resistência a exploração intra-tarefa | pico de `smoothed_delta` ≤ 0.25 | **0.0014** |
| Detecção de shift real de regime | cruza limiar em <100 passos, pico ≥ 1.0 | cruza em ~5 passos, **pico 2.63** |

O detector funciona como um "detector de borda": dispara prontamente no momento da mudança real de regime, depois decai naturalmente de volta a zero conforme a Slow Network converge para a Fast — comportamento esperado e correto, não um defeito.

## 7. Fase 7 — Reintegração: Um Achado Negativo Quantificado e Informativo

Com o detector validado isoladamente, a Fase 7 reintegrou o mecanismo num agente de RL completo (mesma geometria balanceada 2×2 que funcionou melhor na Fase 5, objetivos nos cantos (0,0)/(4,4)), substituindo o `SleepConsolidator` antigo. Foram necessárias três iterações, cada uma movendo o resultado de forma mensurável:

**Tentativa 1 (burst único).** A especificação original pedia que o changepoint disparasse um único burst de n sonhos no momento da borda de subida — replicando a ideia da Fase 4. Resultado: redução de apenas **2.84%**, muito abaixo do limiar de 30%. Diagnóstico: a Região B dura milhares de passos reais após o disparo; um burst único, não importa o tamanho, é diluído pelo volume de treino real subsequente sem nenhuma proteção contínua.

**Tentativa 2 (ensaio contínuo).** Correção: em vez de um burst único, o sistema entra em um "modo de proteção" permanente após o primeiro disparo — a partir daí, todo passo real subsequente vem acompanhado de 1 passo adicional de ensaio (sonhar + treinar), pelo resto do experimento (sem oráculo de fim de tarefa, por desenho). Combinado com uma redução de `beta_slow` em ordens de magnitude (de 0.02 para 0.00005, mantendo a Slow Network relevante por mais tempo), a redução subiu para **17.79%**.

**Tentativa 3 (proporção 4:1).** Aumentando a proporção de ensaio por passo real de 1:1 para 4:1, a redução chegou a **25.09%** — ainda abaixo do limiar de 30%, mas numa trajetória monotonicamente crescente e bem caracterizada: **2.84% → 17.79% → 25.09%**.

Por acordo prévio entre arquiteto e executor, essa foi a última tentativa autorizada nesta fase. O resultado final é um **achado negativo quantificado**: o mecanismo Dual-Weight PCN funciona e protege substancialmente mais conhecimento do que o `SleepConsolidator` original (que não produzia proteção mensurável nenhuma na Fase 5), mas não o suficiente para cruzar o limiar de 30% definido a priori, dentro do orçamento de tentativas estabelecido. Não está determinado se aumentar ainda mais a proporção de ensaio cruzaria o limiar eventualmente, ou se existe um platô — essa é uma pergunta aberta explícita para trabalho futuro (§10), discutida com mais contexto em §8.1.

Nota metodológica: a taxa de sucesso de navegação permaneceu em 0% em todas as condições pós-Região-B (com e sem sono), reconfirmando a lição da Fase 5 de que essa métrica é insensível ao tipo de degradação medido — todas as comparações desta fase usam MSE direto, não taxa de sucesso.

## 8. Fase 8 — Replay Priorizado: o Mecanismo que Cruzou o Limiar

Revisando o resultado da Fase 7, o usuário observou que o cérebro é eficiente não apenas por usar regras locais, mas porque usa atenção e importância para não processar/guardar tudo com peso igual. Essa observação deu origem à terceira rodada de debate (ver [DEBATE_RODADA3.md](DEBATE_RODADA3.md)), que validou a intuição com literatura própria — "synaptic tagging and capture" e replay hipocampal tendencioso por saliência na neurociência; Elastic Weight Consolidation (Kirkpatrick et al., 2017) e Prioritized Experience Replay (Schaul et al., 2015) em ML — e convergiu num mecanismo: substituir a amostragem de sonhos por gaussiana cega (usada em toda a Fase 7) por amostragem proporcional à importância real, medida pela densidade de `visit_count` que o `LearningProgressTracker` já calculava.

**Tentativa 1 (priorização linear pura).** Amostrar diretamente proporcional a `visit_count` produziu **-11.06%** — pior que a gaussiana cega da Fase 7. Investigação direta revelou a causa: 95.7% de todas as visitas de treino caíram num único par (célula, ação); a própria célula-objetivo teve zero visitas (o episódio termina ao alcançá-la). A amostragem colapsou quase inteiramente nesse único par, deixando as outras 19 transições do conjunto de avaliação — incluindo o objetivo — sem nenhuma proteção. Isso não foi um bug: é precisamente o fenômeno de "loss of diversity" que o próprio Schaul et al. (2015) documentou e corrigiu no paper original de Prioritized Experience Replay, usando priorização suavizada em vez de proporcionalidade linear direta — uma correção presente na literatura citada na abertura do debate, mas omitida na primeira especificação.

**Tentativa 2 (priorização suavizada).** Substituindo a amostragem direta por `(visit_count + ε)^α` (com `ε=1.0` garantindo piso mínimo de probabilidade para toda transição, `α=0.5` suavizando sem eliminar a priorização), a concentração no par mais visitado caiu de 96% para 21.45%, e a redução de esquecimento saltou para **64.91%** — mais que o dobro do resultado da Fase 7, muito acima do limiar de 30%.

A captura do snapshot de `visit_count` usado para a amostragem seguiu a mesma disciplina já estabelecida na Fase 7: tirado uma única vez, no instante em que o detector de changepoint da Dual-Weight PCN (Fase 6) dispara pela primeira vez — nenhum oráculo externo de fronteira de tarefa foi introduzido.

### Trajetória completa do problema de consolidação, três gerações

| Fase | Mecanismo | Redução de esquecimento |
|---|---|---|
| 5 | SleepConsolidator (estatística EMA única) | ~0% |
| 7 | Dual-Weight PCN, ensaio contínuo 4:1 | 25.09% |
| 8, tentativa 1 | + Replay priorizado linear | -11.06% |
| 8, tentativa 2 | + Replay priorizado suavizado | **64.91%** |

## 9. Fase 9 — Teste com 3 Regiões: a Previsão Teórica se Confirma

Desde a Rodada 2 do debate, antes de qualquer implementação do Dual-Weight PCN, havia a previsão explícita de que uma única Slow Network (e, por extensão, um único snapshot de importância) se tornaria "um meio-termo cada vez mais borrado" caso o agente enfrentasse mais de duas regiões sequenciais. A Fase 9 testou essa previsão diretamente, estendendo o protocolo da Fase 8 para uma terceira região C (o terceiro canto do grid, objetivo (0,4)), com uma única mudança mínima e justificada no mecanismo: o snapshot de `visit_count` usado para o replay priorizado passa a ser atualizado a cada nova detecção de changepoint (antes, só era capturado na primeira vez), permitindo que o sistema tente proteger cada nova transição sem qualquer oráculo externo de fronteira de tarefa.

**Resultados** (treino sequencial A→B→C, 225 episódios cada):

| Métrica | Sem sono | Com sono | Redução relativa |
|---|---|---|---|
| MSE em A, após B | 0.4256 | 0.1437 | **66.24%** (consistente com a Fase 8) |
| MSE em A, após C | 0.2078 | 0.2436 | **-17.23%** (proteção perdida) |
| MSE em B, após C | 0.0528 | 0.0258 | **51.19%** (tarefa recente protegida) |

A previsão teórica **se confirmou precisamente**: o mecanismo protege com força a tarefa imediatamente anterior à transição mais recente (B, com 51.19% de redução ao entrar em C — na mesma faixa de eficácia da Fase 8), mas a proteção da região A, que havia sido bem-sucedida após a primeira transição (66.24%), **não sobrevive à segunda** — o resultado com sono fica, na prática, pior que sem sono (-17.23%), porque o snapshot de importância, ao ser atualizado para proteger B, descarta inteiramente a informação sobre A. O mecanismo funciona como uma "memória de profundidade 1": protege a tarefa mais recente, não uma pilha de tarefas.

Um efeito colateral notável, não previsto explicitamente mas consistente com o quadro geral: o MSE da própria região B, medido imediatamente após seu próprio treino (antes de C), foi pior com sono ativo (0.3229) do que sem sono (0.0494). Proteger A durante o treino de B compete com a qualidade do aprendizado fresco de B — um primeiro indício quantificado do trade-off plasticidade-vs-estabilidade que a literatura de Complementary Learning Systems já previa em termos qualitativos.

Este resultado não invalida a Fase 8 — delimita precisamente seu alcance: o mecanismo resolve bem o caso de 2 tarefas sequenciais, e falha de forma previsível e bem caracterizada ao ser estendido para 3. Isso aponta diretamente para o Mecanismo B (proteção seletiva de peso, registrado como trabalho futuro na Rodada 3 do debate) como o próximo passo natural para cenários de continual learning com mais de duas tarefas.

## 10. Fase 10 — Mecanismo B (EWC Local Online): Resolve a Profundidade 1, Mas Custa a Plasticidade

A Rodada 4 do debate convergiu numa especificação precisa do Mecanismo B, adaptando EWC Online multi-tarefa (Kirkpatrick et al. 2017; Schwarz et al. 2018) ao proxy de Fisher local já disponível na regra de aprendizado do PCN (`(e_l · x_{l-1})²`, o gradiente local ao quadrado). Para cada peso, três novos estados são mantidos: uma EMA rápida do proxy (`F_ema`), uma Fisher acumulada com decaimento (`F_total`), e uma âncora ponderada por Fisher (`w_anchor`). A cada changepoint (mesmo sinal Fast-vs-Slow do Mecanismo A), a Fisher da tarefa recém-concluída é incorporada com decaimento `gamma`, e a âncora se desloca proporcionalmente ao peso relativo entre a Fisher antiga (decaída) e a nova. Uma penalidade `λ·F_total·(w - w_anchor)`, somada ao update local existente, mantém a regra 100% local — nenhuma dependência cross-peso foi introduzida, preservando a restrição arquitetural central do projeto. A implementação exigiu a primeira modificação histórica em `pcn_core/model.py` (uma única linha aditiva, verificada como numericamente equivalente ao comportamento anterior).

**Resultado principal** (protocolo A→B→C da Fase 9, Mecanismo A + Mecanismo B juntos, `gamma=0.9`, primeira tentativa com `λ=1.0`): a redução relativa em A-após-C, que era **-17.23%** com o Mecanismo A isolado (Fase 9), subiu para **+52.22%** — o Mecanismo B resolveu precisamente o problema de "profundidade 1" que motivou sua criação, confirmando a hipótese da Rodada 4 de que Fisher acumulada por soma (em vez de snapshot substituído) escapa estruturalmente desse limite.

**Mas o resultado tem um custo severo, não endereçado na primeira rodada.** Uma calibração subsequente de `λ` ∈ {0.01, 0.1, 0.2, 0.3, 1.0} revelou que o MSE da própria região B, medido imediatamente após seu treino, degrada em TODOS os valores testados — de 0.3229 (Mecanismo A isolado, já pior que o baseline sem nenhuma proteção) para entre 0.42 e 0.90 com EWC ativo, independentemente do valor de λ. Nenhum valor testado aproximou o MSE de B do baseline sem EWC (0.0494). A proteção de A, nesta calibração, tem como efeito colateral quase congelar a capacidade da rede de aprender bem a tarefa B em primeiro lugar — não apenas de reter B depois (como no efeito da Fase 9), mas de aprendê-la enquanto acontece.

| λ | Redução A-após-C | MSE B-pós-B |
|---|---|---|
| 0 (só Mecanismo A) | -17.23% | 0.3229 |
| 0.01 | +7.82% | 0.4173 |
| 0.1 | +48.05% | 0.8973 |
| 1.0 | +52.21% | 0.7402 |

A Fase 10 é, portanto, um sucesso parcial honesto: resolve exatamente o problema que foi desenhada para resolver (profundidade 1 em A), mas troca esse ganho por um trade-off plasticidade-vs-estabilidade mais severo do que o observado até aqui — não um ajuste fino de hiperparâmetro, e sim uma limitação estrutural da penalidade global e constante proposta. O trabalho futuro mais direto é tornar `λ` adaptativo por peso (ex: com clipping, ou modulado pelo próprio `F_total` daquele peso), em vez de um escalar único aplicado uniformemente.

## 11. Discussão

O resultado mais importante deste trabalho não é um número de acurácia — é a **confirmação empírica, em código executável, de uma limitação teórica identificada antes de qualquer implementação**: o debate original registrou explicitamente, na síntese final, que "replay generativo tipicamente depende de um sinal de fronteira de tarefa/contexto" e que isso era um "problema aberto de task-free continual learning". A Fase 5 não apenas confirmou essa previsão — ela revelou o *mecanismo exato* pelo qual a falta desse sinal quebra o sistema: a estatística de rastreamento do consolidador de sono, por precisar se adaptar rápido o suficiente para ser útil durante o aprendizado normal, inevitavelmente "esquece" o que deveria proteger assim que uma nova distribuição de experiência começa a dominar.

Isso sugere que **task-free continual learning via estatísticas de rastreamento simples (EMA) é insuficiente**: esses rastreadores não distinguem "a distribuição mudou porque chegou uma nova tarefa relevante" de "a distribuição mudou porque o ruído natural do ambiente variou" — ambos produzem o mesmo sintoma (adaptação rápida do obs_mean/obs_std), mas só o primeiro caso deveria disparar proteção agressiva do conhecimento antigo.

Um segundo achado relevante, de natureza metodológica: **taxa de sucesso em tarefas de navegação é uma métrica enganosa para detectar esquecimento**, porque tarefas com estrutura de recompensa monotônica (like "mais perto é melhor") preservam a ordenação relativa de qualidade das ações mesmo quando a magnitude das previsões degrada completamente. Isso tem implicações para qualquer avaliação de continual learning em RL: métricas de sucesso/recompensa podem mascarar degradação real do modelo subjacente, que só se torna visível em métricas diretas de erro de predição.

### 11.1 Progresso ou parede já esperada? As duas coisas, mas não da mesma forma

Vale distinguir explicitamente dois tipos de resultado negativo que este trabalho produziu, porque são qualitativamente diferentes.

A Fase 5 encontrou uma **parede estrutural**: nenhuma calibração de hiperparâmetro (geometria da tarefa, pesos de exploração, constantes de tempo) produziu proteção mensurável contra esquecimento — os resultados oscilaram entre nulos e levemente negativos (-0.02%) independentemente do ajuste. Isso é consistente com um mecanismo que está, na sua essência, mal-adequado ao problema: uma única estatística EMA não pode representar simultaneamente "o que é relevante agora" e "o que era relevante antes", e nenhuma quantidade de ajuste fino resolve essa contradição interna. Essa parede já havia sido prevista — não em detalhe, mas em princípio — na síntese do debate original, antes de qualquer linha de código.

A Fase 7, com o mecanismo substituto (Dual-Weight PCN), encontrou algo diferente: um **sinal de progresso real, ainda insuficiente**. Cada correção de design (não apenas ajuste de hiperparâmetro, mas mudanças estruturais pequenas e bem-fundamentadas — de burst único para ensaio contínuo, de uma proporção 1:1 para 4:1) moveu o resultado substancialmente e na direção prevista: 2.84% → 17.79% → 25.09%. Uma curva assim — monotonicamente crescente, com incrementos de magnitude comparável a cada mudança — é o padrão esperado de um mecanismo que funciona e está subdimensionado, não de um mecanismo mal-adequado. Não cruzamos o limiar de 30% dentro do orçamento de tentativas definido a priori, mas não temos evidência de que exista um platô abaixo dele — essa distinção importa para decidir o que fazer a seguir: insistir em mais dosagem (Fase 7) é uma aposta razoável; insistir em mais calibração do mecanismo antigo (Fase 5) já não seria.

Em resumo: a arquitetura como um todo esteve evoluindo dentro de um limite teórico que já era conhecido (task-free continual learning), e a *qualidade* da aproximação a esse limite mudou de categoria a cada geração do mecanismo de consolidação. A Fase 8 completou a trajetória de 2 tarefas: não foi mais dosagem do mesmo mecanismo que cruzou o limiar de 30%, foi uma troca qualitativa — de "quanto repetir" para "o que importa repetir". A Fase 9 (§9), por sua vez, mostrou exatamente onde essa vitória tem fronteira: estendido para 3 tarefas sequenciais, o mesmo mecanismo que resolveu 2 volta a encontrar a parede teórica original — não por falta de calibração, mas porque um único snapshot de importância, tal como uma única Slow Network, só tem "espaço" para proteger uma tarefa por vez. A previsão feita na Rodada 2 do debate, antes de qualquer código, descreveu esse limite com precisão. O projeto demonstra, portanto, três categorias de resultado, não duas: parede estrutural (Fase 5), progresso subdimensionado (Fase 7), vitória qualitativa dentro de um alcance limitado (Fases 8-9) — e a fronteira exata desse alcance (2 tarefas, não 3) está agora medida, não apenas prevista.

## 12. Limitações

- **Escala**: todos os experimentos usam redes com no máximo 136 parâmetros de entrada equivalentes (camada `[64,32,16,10]` na Fase 1) e um ambiente de 25 estados discretos (GridWorld 5×5). A equivalência PCN↔backprop está provada na literatura para redes feedforward/CNN/RNN de porte convencional; nada neste trabalho testa ou sugere que isso se mantenha em escalas comparáveis a modelos de linguagem.
- **Nenhuma validação de hardware real**: a alegação de eficiência energética do substrato local é arquitetural (compatibilidade conceitual com regras event-driven), não medida em watts/joules em silício neuromórfico real.
- **Ambiente trivial**: GridWorld 5×5 totalmente observável, sem física contínua, sem hierarquia de sub-objetivos, sem múltiplos agentes.
- **Fronteira de tarefa**: a Fase 4 (validada) depende de conhecer o momento exato de transição entre tarefas de antemão. A Fase 5 mostrou que remover essa dependência ingenuamente quebra o mecanismo original. O mecanismo final que funcionou (Fase 8) não depende de um oráculo externo — usa o detector de changepoint da Dual-Weight PCN (Fase 6) — mas ainda depende de um proxy de importância (`visit_count`) cuja validade foi testada só para uma única transição A→B, não para sequências mais longas.
- **Nenhum teste de linguagem ou raciocínio simbólico** foi conduzido — o escopo deste trabalho é estritamente sensório-motor.

## 13. Trabalhos Futuros

1. **`λ` adaptativo por peso no Mecanismo B** — o passo mais direto e urgente: a Fase 10 mostrou que um `λ` escalar e constante resolve a profundidade 1 em A mas quebra a plasticidade de aprendizagem de B em qualquer valor testado (0.01 a 1.0). Modular `λ` pelo próprio `F_total` daquele peso (ex: com clipping ou normalização), em vez de um multiplicador único global, é a hipótese mais promissora não testada.
2. **Memória de profundidade >1** para o snapshot de importância do Mecanismo A — por exemplo, manter K snapshots (não só o mais recente) e amostrar sonhos de todos eles; alternativa ao Mecanismo B que não foi testada, e que evita por completo o trade-off de plasticidade observado na Fase 10 (já que não penaliza pesos, só redistribui a amostragem de sonhos).
3. **Investigar formalmente o trade-off plasticidade-vs-estabilidade** quantificado nas Fases 9 e 10 (Mecanismo A isolado já degrada o aprendizado de B; Mecanismo B agrava isso em todos os `λ` testados) — até que ponto isso é um limite estrutural de qualquer proteção baseada em penalização/replay, e existe uma família de mecanismos que escapa dele por desenho?
4. **Generalizar o proxy de importância** além de `visit_count` — por exemplo, combinar frequência de visita com uma medida de "quão necessário" cada transição é para o comportamento ótimo, para cenários onde a política de navegação não produz naturalmente uma distribuição de visitas informativa.
5. **Escalar a rede e o ambiente** gradualmente (grid maior, observação parcial, ações contínuas) para identificar em qual ponto a equivalência PCN↔backprop começa a degradar, se degradar.
6. **Medição real de eficiência energética** em hardware neuromórfico (ex: simulação em Loihi/NorthPole, ou ao menos contagem de operações event-driven vs densas).
7. **Investigar métricas de avaliação de continual learning que não sejam enganadas por estrutura de recompensa monotônica** — generalizar a lição da Fase 5 para benchmarks de RL continual mais amplos.

## 14. Conclusão

Este trabalho validou isoladamente os quatro mecanismos propostos no debate original — aprendizado local equivalente a backprop, unificação de percepção e ação via Active Inference, motivação intrínseca resistente a ruído via Learning Progress, e consolidação sem buffer via sono generativo — com evidência experimental concreta e reproduzível em cada caso. A primeira tentativa de integração *task-free* (Fase 5) revelou, com rigor empírico, exatamente a limitação que a concepção teórica da arquitetura já havia identificado como não resolvida: a dependência de um sinal de fronteira de tarefa para que a consolidação funcione. Uma segunda rodada de debate produziu um mecanismo substituto (Dual-Weight PCN) validado isoladamente com margem confortável (Fase 6); sua reintegração (Fase 7) produziu proteção real mas insuficiente (25.09% de 30% exigidos). Uma terceira rodada de debate, iniciada por uma observação do usuário humano sobre como o cérebro usa atenção e importância para evitar processar tudo igualmente, levou à troca da amostragem genérica de sonhos por replay priorizado por importância real — e, após uma autocorreção rigorosa de uma primeira versão que piorou o resultado por perda de diversidade, produziu **64.91% de redução**, cruzando o limiar com folga (Fase 8). O resultado demonstra que a arquitetura Developmental PCN, quando seus quatro pilares são corretamente integrados com um mecanismo de consolidação informado por importância — não apenas volume de repetição —, resolve o problema de esquecimento catastrófico *task-free* para o caso de duas tarefas sequenciais. Uma quarta investigação (Fase 9), estendendo o mesmo mecanismo para três tarefas, mediu precisamente onde esse sucesso termina: a proteção funciona como uma memória de profundidade 1, protegendo fortemente a tarefa mais recente mas perdendo a mais antiga a cada nova transição — exatamente a limitação que a segunda rodada de debate havia previsto, agora quantificada em números reproduzíveis em vez de apenas antecipada em princípio. Uma quarta rodada de debate projetou o Mecanismo B (EWC Local Online) especificamente para resolver essa limitação, mantendo a restrição arquitetural de localidade; sua implementação e teste (Fase 10) confirmou que a hipótese estava certa — Fisher acumulada por soma escapa da profundidade 1, elevando a redução em A-após-C de -17.23% para +52.22% — mas a calibração subsequente de `λ` revelou que esse ganho específico tem um custo que nenhum valor testado resolveu: a plasticidade da rede para aprender a tarefa seguinte degrada sob a penalidade, trocando um problema bem compreendido (memória superficial) por outro, estruturalmente mais difícil (rigidez excessiva). A trajetória completa (cinco gerações de investigação, quatro rodadas de debate, três intuições ou decisões humanas decisivas) é, em si, o resultado mais relevante deste trabalho: mostra que um limite teórico bem caracterizado pode ser superado — não por mais escala do mesmo mecanismo, mas por uma mudança qualitativa de princípio —, que a fronteira exata de cada sucesso pode ser medida com a mesma disciplina usada para alcançá-lo, e que resolver uma limitação pode revelar a próxima, numa cadeia que a honestidade metodológica torna visível em vez de esconder. A colaboração entre intuição humana e rigor técnico de múltiplos agentes de IA, com verificação independente em cada etapa — incluindo a recusa de aceitar "APROVADO" sem examinar o custo escondido atrás do número que bateu a meta —, foi o que permitiu tanto identificar e corrigir as mudanças que funcionaram quanto caracterizar honestamente onde cada uma para de funcionar.

## Referências

- Adams, R. P., & MacKay, D. J. C. (2007). *Bayesian Online Changepoint Detection*. arXiv:0710.3742.
- Aljundi, R., Kelchtermans, K., & Tuytelaars, T. (2019). *Task-Free Continual Learning*. CVPR 2019.
- Behrouz, A., Zhong, P., & Mirrokni, V. (2024). *Titans: Learning to Memorize at Test Time*. arXiv:2501.00663.
- Burda, Y., Edwards, H., Pathak, D., Storkey, A., Darrell, T., & Efros, A. (2018). *Large-Scale Study of Curiosity-Driven Learning*. arXiv:1808.04355.
- De Lange, M., & Tuytelaars, T. (2021). *Continual Prototype Evolution: Learning Online from Non-Stationary Data Streams*. ICCV 2021.
- Frey, U., & Morris, R. G. M. (1997). *Synaptic tagging and long-term potentiation*. Nature, 385, 533–536.
- Friston, K., et al. (2017). *Uncertainty, epistemics and active inference*. Journal of the Royal Society Interface.
- Friston, K., et al. (2018). *Bayesian model reduction*. arXiv:1805.07092.
- Kirkpatrick, J., et al. (2017). *Overcoming catastrophic forgetting in neural networks*. PNAS, 114(13).
- Kumaran, D., Hassabis, D., & McClelland, J. L. (2016). *What Learning Systems do Intelligent Agents Need? Complementary Learning Systems Theory Updated*. Trends in Cognitive Sciences.
- LeCun, Y. (2024). *Objective-Driven AI: Towards AI systems that can learn, remember, reason, and plan*.
- McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). *Why there are complementary learning systems in the hippocampus and neocortex*. Psychological Review.
- Millidge, B., Song, Y., et al. (2021). *Predictive Coding Can Do Exact Backpropagation on Convolutional and Recurrent Neural Networks*. arXiv:2103.03725.
- Oudeyer, P.-Y., & Kaplan, F. (2007). *Intrinsic Motivation, Curiosity, and Learning: Theory and Applications in Educational Technologies*.
- Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015). *Prioritized Experience Replay*. arXiv:1511.05952.
- Schwarz, J., et al. (2018). *Progress & Compress: A scalable framework for continual learning*. ICML 2018.
- van de Ven, G. M., Siegelmann, H. T., & Tolias, A. S. (2020). *Brain-inspired replay for continual learning with artificial neural networks*. Nature Communications, 11, 4069.
- Zenke, F., Poole, B., & Ganguli, S. (2017). *Continual Learning Through Synaptic Intelligence*. ICML 2017.
- Whittington, J. C. R., & Bogacz, R. (2017). *An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity*. Neural Computation.
