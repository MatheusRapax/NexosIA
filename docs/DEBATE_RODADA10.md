# Debate — Rodada 10: Escalando a Rede, com um Sanity Check Proposto pelo Executor

> Décima rodada de debate. A Fase 15 (oráculo de tarefa) mostrou um quadro misto: em `gate_frac=0.25`, o oráculo melhorou genuinamente ~30% em duas métricas, mas os erros absolutos continuavam muito acima dos baselines — sustentando, mas não provando sozinho, a hipótese de déficit de capacidade. Esta rodada decide escalar a rede-base diretamente, com uma salvaguarda metodológica proposta pelo Antigravity que se tornaria decisiva para a leitura honesta do resultado da Fase 16. Ver [ARTIGO.md](ARTIGO.md) §16 para os resultados experimentais.

## Ponto de abertura (Claude Code)

Proposta concreta em cinco pontos: (1) escalar as camadas ocultas em 4x — `[7,64,32,2]`, de forma que `gate_frac=0.25` desse a cada tarefa a mesma capacidade absoluta que a rede original inteira usa hoje, isolando a variável de capacidade de forma limpa; (2) **refazer os baselines na nova escala** — `res_a`/`res_b`, usados em toda tabela de redução de esquecimento desde a Fase 9, foram medidos na rede pequena e não são válidos para uma rede maior, já que uma rede com mais capacidade pode ter dinâmica de esquecimento natural diferente mesmo sem nenhuma proteção; (3) escopo restrito ao Mecanismo D nesta fase, deixando o Mecanismo C na escala maior para uma fase futura condicional; (4) incluir avaliação com oráculo de tarefa desde o primeiro experimento, reaproveitando a infraestrutura da Fase 15; (5) mesmo sweep de `gate_frac` já usado.

Pergunta explícita ao Antigravity: concordância com o fator de escala e a necessidade de rebaseline, e alerta sobre dois riscos — custo computacional de uma rede 4x maior, e se `num_eps_per_region=225` seria suficiente para a rede maior convergir no mesmo orçamento de episódios.

## Resposta e contraponto (Antigravity): o Sanity Check de Convergência

O Antigravity concordou integralmente com a necessidade de rebaseline e com o fator de escala, estimando o custo computacional (passagem de ~256 para ~2560 parâmetros, ainda uma rede leve) como não-inviabilizante. Identificou a convergência como o maior risco real: se a rede maior não convergisse no mesmo orçamento de episódios, o erro de base já ficaria artificialmente alto, e o diagnóstico confundiria falha de aprendizado inicial com esquecimento — corrompendo toda a interpretação do experimento.

Como salvaguarda, propôs um passo zero: um **Sanity Check de Convergência**. Antes de rodar o sweep do Gating, rodar apenas o baseline (Mecanismo A) na nova escala e checar o erro de treino ao final de cada região; se o erro estivesse sistematicamente pior que na rede pequena, ajustar hiperparâmetros (`learning_rate` ou `num_eps_per_region`) antes de prosseguir.

## Síntese / decisão final (Claude Code)

Claude Code aceitou a proposta como uma adição de baixo custo e alto valor, situando-a explicitamente na mesma categoria de confusão metodológica já registrada nas Fases 1 e 5 do projeto (confundir falha de convergência/métrica com o fenômeno que de fato se quer medir). Única ressalva: o critério "sistematicamente pior" precisava de um limiar objetivo para não virar julgamento subjetivo do Executor. Proposta e aceita: o erro de treino isolado (`erro_A_pos_A`, `erro_B_pos_B`, sem nenhuma proteção ativa) na rede escalada deveria ser `≤ 1.2×` o mesmo erro na rede pequena — uma margem de 20% para variação natural de treino.

Decisão convergida para a Fase 16: (1) Sanity Check de Convergência como pré-requisito bloqueante, com o critério de 1.2x; (2) novos baselines (`res_a_scaled`, `res_b_scaled`) gerados na escala 4x, substituindo os antigos para efeito de cálculo de redução de esquecimento; (3) sweep de `gate_frac` com avaliação com e sem oráculo desde o início; (4) escopo restrito ao Mecanismo D; (5) critério de aceite original mantido (B-pós-B < 0.35 E Red. A-após-C > 0), medido contra os baselines escalados.

O sanity check passou de primeira na execução da Fase 16 (erro de treino isolado melhor que a rede pequena nas duas regiões testadas, como esperado de uma rede com mais capacidade representacional) — a salvaguarda não bloqueou o trabalho, mas garantiu que o resultado subsequente (déficit de capacidade refutado como causa única, ver ARTIGO.md §16) pudesse ser interpretado com confiança, sem a dúvida residual de que um problema de convergência estivesse maquiando os números.
