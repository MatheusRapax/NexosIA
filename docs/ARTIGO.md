# Developmental PCN: Aprendizado Local, Motivação Intrínseca e Consolidação Sem Buffer — Prova de Conceito e Limites Empíricos

**Tipo:** Relatório técnico-científico de prova de conceito (proof-of-concept), com achado negativo documentado.
**Data:** Setembro–Outubro de 2026.
**Autores (processo):** Arquitetura concebida em debate estruturado entre dois agentes de IA (Claude Code e Antigravity, via Maestri); implementação conduzida sob fluxo Arquiteto/Executor, com verificação independente de cada entrega.

## Resumo

Modelos de linguagem de grande escala (LLMs) dependem de computação massiva porque separam memória de longo prazo (pesos, congelados após o treino) de memória de trabalho (contexto, descartada a cada sessão), forçando recomputação bruta via atenção a cada inferência. Este trabalho implementa e testa empiricamente uma arquitetura alternativa — a **Developmental PCN** — que unifica memória e cômputo num único substrato de pesos plásticos, aprendendo por regras estritamente locais (sem backpropagation global). A arquitetura tem quatro pilares: (1) um motor de Predictive Coding Network (PCN) matematicamente equivalente a backprop mas implementável com message passing local; (2) Active Inference como princípio unificador de percepção e ação; (3) Learning Progress como motivação intrínseca, substituindo erro de predição bruto; (4) consolidação via "sono" com replay generativo implícito e poda por precisão, sem buffer de experiências reais. Validamos cada pilar isoladamente com sucesso (Fases 1–4), obtendo: paridade com um baseline de backprop em classificação (-0.56 pontos percentuais); navegação bem-sucedida num ambiente simulado usando o mesmo substrato para memória e ação (100% de sucesso); resistência empírica ao "Noisy TV problem" via Learning Progress; e redução de 66.57% no esquecimento catastrófico via sono generativo. A tentativa de integrar os quatro pilares num único agente contínuo (Fase 5) **falhou em quatro iterações sucessivas**, revelando um limite genuíno: a ausência de um sinal de fronteira de tarefa (*task-free continual learning*) degrada tanto a convergência do modelo de mundo quanto a eficácia do mecanismo de consolidação. Esse achado negativo confirma, empiricamente, uma limitação que havia sido antecipada — mas não resolvida — na concepção teórica da arquitetura.

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

## 6. Discussão

O resultado mais importante deste trabalho não é um número de acurácia — é a **confirmação empírica, em código executável, de uma limitação teórica identificada antes de qualquer implementação**: o debate original registrou explicitamente, na síntese final, que "replay generativo tipicamente depende de um sinal de fronteira de tarefa/contexto" e que isso era um "problema aberto de task-free continual learning". A Fase 5 não apenas confirmou essa previsão — ela revelou o *mecanismo exato* pelo qual a falta desse sinal quebra o sistema: a estatística de rastreamento do consolidador de sono, por precisar se adaptar rápido o suficiente para ser útil durante o aprendizado normal, inevitavelmente "esquece" o que deveria proteger assim que uma nova distribuição de experiência começa a dominar.

Isso sugere que **task-free continual learning via estatísticas de rastreamento simples (EMA) é insuficiente**: esses rastreadores não distinguem "a distribuição mudou porque chegou uma nova tarefa relevante" de "a distribuição mudou porque o ruído natural do ambiente variou" — ambos produzem o mesmo sintoma (adaptação rápida do obs_mean/obs_std), mas só o primeiro caso deveria disparar proteção agressiva do conhecimento antigo.

Um segundo achado relevante, de natureza metodológica: **taxa de sucesso em tarefas de navegação é uma métrica enganosa para detectar esquecimento**, porque tarefas com estrutura de recompensa monotônica (like "mais perto é melhor") preservam a ordenação relativa de qualidade das ações mesmo quando a magnitude das previsões degrada completamente. Isso tem implicações para qualquer avaliação de continual learning em RL: métricas de sucesso/recompensa podem mascarar degradação real do modelo subjacente, que só se torna visível em métricas diretas de erro de predição.

## 7. Limitações

- **Escala**: todos os experimentos usam redes com no máximo 136 parâmetros de entrada equivalentes (camada `[64,32,16,10]` na Fase 1) e um ambiente de 25 estados discretos (GridWorld 5×5). A equivalência PCN↔backprop está provada na literatura para redes feedforward/CNN/RNN de porte convencional; nada neste trabalho testa ou sugere que isso se mantenha em escalas comparáveis a modelos de linguagem.
- **Nenhuma validação de hardware real**: a alegação de eficiência energética do substrato local é arquitetural (compatibilidade conceitual com regras event-driven), não medida em watts/joules em silício neuromórfico real.
- **Ambiente trivial**: GridWorld 5×5 totalmente observável, sem física contínua, sem hierarquia de sub-objetivos, sem múltiplos agentes.
- **Fronteira de tarefa**: a Fase 4 (validada) depende de conhecer o momento exato de transição entre tarefas; a Fase 5 (não validada) mostrou que remover essa dependência quebra o mecanismo nas condições testadas.
- **Nenhum teste de linguagem ou raciocínio simbólico** foi conduzido — o escopo deste trabalho é estritamente sensório-motor.

## 8. Trabalhos Futuros

1. **Detecção de mudança de distribuição mais robusta que EMA simples** para disparo de consolidação — por exemplo, testes estatísticos de divergência (KL, variância de janela deslizante) capazes de distinguir ruído de mudança de regime.
2. **Resolução de task-free continual learning** com um sinal de "surpresa estrutural" (não apenas erro instantâneo) que dispare sono de forma adaptativa, em vez de periódica por contagem fixa de episódios.
3. **Escalar a rede e o ambiente** gradualmente (grid maior, observação parcial, ações contínuas) para identificar em qual ponto a equivalência PCN↔backprop começa a degradar, se degradar.
4. **Medição real de eficiência energética** em hardware neuromórfico (ex: simulação em Loihi/NorthPole, ou ao menos contagem de operações event-driven vs densas).
5. **Investigar métricas de avaliação de continual learning que não sejam enganadas por estrutura de recompensa monotônica** — generalizar a lição da Fase 5 para benchmarks de RL continual mais amplos.

## 9. Conclusão

Este trabalho validou isoladamente os quatro mecanismos propostos no debate original — aprendizado local equivalente a backprop, unificação de percepção e ação via Active Inference, motivação intrínseca resistente a ruído via Learning Progress, e consolidação sem buffer via sono generativo — com evidência experimental concreta e reproduzível em cada caso. A tentativa de integração revelou, com rigor empírico, exatamente a limitação que a concepção teórica da arquitetura já havia identificado como não resolvida: a dependência de um sinal de fronteira de tarefa para que a consolidação funcione. Isso não invalida a arquitetura — delimita precisamente a fronteira do que ela resolve hoje, e aponta o problema concreto e bem caracterizado que uma continuação deste trabalho precisaria atacar.

## Referências

- Behrouz, A., Zhong, P., & Mirrokni, V. (2024). *Titans: Learning to Memorize at Test Time*. arXiv:2501.00663.
- Burda, Y., Edwards, H., Pathak, D., Storkey, A., Darrell, T., & Efros, A. (2018). *Large-Scale Study of Curiosity-Driven Learning*. arXiv:1808.04355.
- Friston, K., et al. (2017). *Uncertainty, epistemics and active inference*. Journal of the Royal Society Interface.
- Friston, K., et al. (2018). *Bayesian model reduction*. arXiv:1805.07092.
- Kumaran, D., Hassabis, D., & McClelland, J. L. (2016). *What Learning Systems do Intelligent Agents Need? Complementary Learning Systems Theory Updated*. Trends in Cognitive Sciences.
- LeCun, Y. (2024). *Objective-Driven AI: Towards AI systems that can learn, remember, reason, and plan*.
- McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). *Why there are complementary learning systems in the hippocampus and neocortex*. Psychological Review.
- Millidge, B., Song, Y., et al. (2021). *Predictive Coding Can Do Exact Backpropagation on Convolutional and Recurrent Neural Networks*. arXiv:2103.03725.
- Oudeyer, P.-Y., & Kaplan, F. (2007). *Intrinsic Motivation, Curiosity, and Learning: Theory and Applications in Educational Technologies*.
- van de Ven, G. M., Siegelmann, H. T., & Tolias, A. S. (2020). *Brain-inspired replay for continual learning with artificial neural networks*. Nature Communications, 11, 4069.
- Whittington, J. C. R., & Bogacz, R. (2017). *An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity*. Neural Computation.
