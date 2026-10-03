# Debate — Rodada 3: Atenção, Importância, e a Virada do Projeto

> Terceira rodada de debate, iniciada por uma observação do próprio usuário (não dos agentes): o cérebro é eficiente não só por usar regras locais, mas porque usa atenção/importância para não precisar guardar ou processar tudo igualmente. Esta rodada produziu o mecanismo que finalmente cruzou o limiar de 30% de redução de esquecimento, depois de duas gerações anteriores (Fases 5 e 7) terem ficado abaixo dele. Ver [ARTIGO.md](ARTIGO.md) §12 para os resultados experimentais (Fase 8).

## Origem

O usuário, revisando o resultado da Fase 7 (25.09% de redução, obtido por repetição bruta — mais volume de ensaio genérico), observou que o cérebro não guarda/processa tudo com peso igual — ele sempre trabalha em cima de importância e atenção. Essa observação foi levada ao debate como hipótese a investigar, não como solução pronta.

## Validação da intuição com literatura

A intuição tinha fundamentação direta em dois campos:

**Neurociência**: "Synaptic tagging and capture" (Frey & Morris, 1997) mostra que uma sinapse só é marcada para consolidação de longo prazo se a ativação for saliente o suficiente. O replay hipocampal durante o sono é sistematicamente tendencioso — prioriza experiências novas, recompensadoras ou repetidas, não reproduz tudo com peso igual ("The role of experience in prioritizing hippocampal replay", Nature Communications, 2023).

**Machine Learning**: a mesma ideia foi formalizada de duas formas: **Elastic Weight Consolidation** (Kirkpatrick et al., 2017) protege pesos importantes via informação de Fisher; **Prioritized Experience Replay** (Schaul et al., 2015) amostra experiências para repetição proporcionalmente à sua utilidade (TD-error), não uniformemente.

## Dois mecanismos debatidos

- **Mecanismo A (Replay Priorizado)**: reusar o `LearningProgressTracker` já existente como distribuição de amostragem dos sonhos, proporcional a `visit_count`, em vez de uma gaussiana cega sobre toda a região.
- **Mecanismo B (EWC local)**: proteger pesos individuais via um proxy local de informação de Fisher (gradiente ao quadrado), com ancoragem.

O Antigravity apontou dois furos reais em B: (1) a formulação inicial confundia erro acumulado com gradiente — corrigido para `(e_l · x_{l-1})²`, a Fisher empírica correta, já calculada implicitamente pela regra local existente; (2) proteger só reduzindo a taxa de aprendizado (sem âncora) sofreria do mesmo drift-lento-mas-eventual que já quebrou o `beta_slow` da Dual-Weight em horizontes longos — proteção real exigiria uma âncora, custando um terceiro valor de estado por peso.

**Decisão**: implementar A primeiro (mais barato, menor risco); registrar B-corrigido como trabalho futuro, não descartado.

## A surpresa: a primeira versão de A piorou o resultado

A implementação inicial de A (amostragem linear proporcional a `visit_count`) produziu **-11.06%** — pior que a gaussiana cega da Fase 7. Investigação direta revelou a causa: 95.7% de todas as visitas da Região A durante o treino caíram num único par (célula, ação); a própria célula-objetivo teve zero visitas (o episódio termina ao alcançá-la). Amostragem proporcional pura concentrou quase todos os sonhos nesse único par, deixando as outras 19 transições do conjunto de avaliação — incluindo o objetivo — sem proteção.

Isso não foi um bug de implementação. É exatamente o fenômeno de **"loss of diversity"** que o próprio artigo do Prioritized Experience Replay (Schaul et al., 2015) documentou e corrigiu, usando priorização suavizada (expoente `alpha < 1` sobre a prioridade) em vez de proporcionalidade linear direta — uma correção que havia sido citada na literatura de abertura do debate, mas não incluída na primeira especificação.

## A correção e o resultado final

Trocando a amostragem direta por `(visit_count + ε)^α` (com `ε=1.0` garantindo piso mínimo de probabilidade para toda transição, `α=0.5` suavizando a dominância do par mais visitado sem eliminar a priorização), a concentração no par mais visitado caiu de 96% para 21.45%, e a redução de esquecimento saltou para **64.91%** — mais que o dobro do resultado da Fase 7, e muito acima do limiar de 30% definido desde a Fase 4.

## Trajetória completa do projeto (Fases 5, 7 e 8)

| Fase | Mecanismo | Redução de esquecimento |
|---|---|---|
| 5 | SleepConsolidator (EMA única) | ~0% (sem proteção mensurável) |
| 7, tentativa 1 | Dual-Weight, burst único | 2.84% |
| 7, tentativa 2 | Dual-Weight, ensaio contínuo 1:1 | 17.79% |
| 7, tentativa 3 | Dual-Weight, ensaio contínuo 4:1 | 25.09% |
| 8, tentativa 1 | + Replay priorizado (linear puro) | -11.06% (pior — loss of diversity) |
| 8, tentativa 2 | + Replay priorizado (suavizado) | **64.91%** ✅ |

A observação do usuário sobre atenção e importância no cérebro não foi apenas uma analogia bonita — foi o insight que destravou, em duas rodadas de debate e uma correção precisa, o problema que duas gerações completas de mecanismo (Fases 5 e 7) não conseguiram resolver.
