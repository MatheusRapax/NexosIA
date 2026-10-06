# Debate — Rodada 7: Da Penalidade Suave à Máscara Dura

> Sétima rodada de debate. A Fase 12 (`c_layer=max`) refutou a hipótese da Rodada 6: eliminou o modo de falha específico (fração de `ratio>0.5` caiu a 0%) mas não resolveu o trade-off de plasticidade — nenhum `lam` satisfez o critério de aceite. Esta rodada abandona a calibração de normalização/`lam` e questiona a premissa estrutural do próprio Mecanismo B, convergindo no Mecanismo C (Máscara Dura + Reset). Ver [ARTIGO.md](ARTIGO.md) §13 para os resultados experimentais (Fase 13).
>
> **Nota de proveniência**: o registro original desta rodada (nota de debate interna do projeto) termina de forma abrupta, no meio da frase final da proposta do Antigravity — a síntese de fechamento explícita nunca foi escrita antes da nota ser sobrescrita por trabalho seguinte. A reconstrução abaixo preserva tudo o que foi efetivamente registrado; a "decisão" ao final é inferida do que a Fase 13 de fato implementou (que adotou a proposta do Antigravity quase literalmente), não de uma frase de síntese que exista no registro original.

## Diagnóstico de fundo (Claude Code)

Depois de duas rodadas inteiras (5 e 6) varrendo normalização de `c_layer` e sweep de `lam` sem resolver o trade-off, Claude Code propôs que o problema não era mais de calibração — era estrutural: penalidades do tipo EWC, aplicadas sobre os MESMOS pesos compartilhados entre tarefas, impõem um trade-off global incontornável por desenho. Não existe `lam` que proteja A sem sufocar a plasticidade de B, porque ambas competem pela mesma capacidade fixa. Mudar a normalização de `F_total` muda só a FORMA da penalidade, não remove o teto arquitetural.

## Pesquisa de literatura trazida para a rodada

- **Structural Synaptic Plasticity Has High Memory Capacity...** (Knoblauch et al., PLoS ONE 2014, PMC4032253): plasticidade ESTRUTURAL (criar/remover sinapses) tem capacidade de memória muito maior que plasticidade de PESO (ajustar magnitude) — o gargalo não é só quanto se protege um peso, é se a memória usa capacidade física separada.
- **NISPA** (Gurbuz & Dovrolis, ICML 2022, arXiv:2206.09117): aloca dinamicamente unidades/conexões "estáveis" (congeladas, dedicadas a tarefas antigas) vs. "plásticas" (livres para tarefas novas), crescendo novas conexões quando a capacidade livre se esgota.
- **Progressive Neural Networks / PackNet**: arquiteturas que criam colunas/sub-redes novas por tarefa (crescimento estrutural) em vez de reescrever pesos antigos.
- **ANPyC** (Adversarial Neural Pruning + synaptic Consolidation): poda (LTD) remove parâmetros irrelevantes à tarefa, consolidação (LTP) fortalece os relevantes — par pruning+consolidation, não só penalidade.
- Uma revisão recente (2024, arXiv:2405.16922) conectando *tagging-and-capture* biológico (consolidação seletiva, não global) a continual learning artificial.

## Proposta inicial: crescimento estrutural

Migrar de "Mecanismo B = penalidade suave" para "Mecanismo C = máscara dura de consolidação + crescimento estrutural": pesos/unidades com `F_total` alto após A ficam CONGELADOS (máscara binária, não penalidade contínua — análogo a uma espinha dendrítica estabilizada); a Tarefa B é forçada a usar capacidade LIVRE (análogo a brotamento de novos ramos/sinapses); se a capacidade livre for insuficiente, CRESCER (adicionar unidades/conexões) em vez de forçar B a competir pelos pesos de A.

Riscos levantados pelo próprio Claude Code antes de passar a palavra ao Antigravity: crescimento ilimitado não escala sem um orçamento de capacidade; a granularidade da máscara (peso individual, bloco por camada, ou unidade/neurônio inteiro) muda tudo; a máscara dura pode ser mais simples de implementar que o EWC atual, valendo testar como hipótese alternativa antes de mais sweeps de `lam`.

## Contraponto do Antigravity: concordância no diagnóstico, discordância no remédio

O Antigravity concordou integralmente com o diagnóstico estrutural — penalidades suaves em pesos compartilhados geram um trade-off global incontornável — mas levantou três riscos concretos específicos do crescimento estrutural aplicado ao PCN:

1. **Interferência no forward**: pesos congelados de A continuariam operando no forward de B; a Tarefa B gastaria sua capacidade livre só para anular esse sinal residual (um risco que, de forma notável, reapareceria mais tarde como achado real de diagnóstico na Fase 14 — não só para crescimento, mas para o próprio Mecanismo C de máscara dura).
2. **Complexidade arquitetural**: o PCN requer nós de valor e erro emparelhados; crescimento dinâmico de nós e conexões em tempo de execução é arriscado e quebra a estabilidade da formulação de energia.
3. **Esparsidade**: máscara por peso (sinapse) gera redes "buracadas", difíceis de otimizar para novas tarefas.

Como alternativa mais simples, que evita os três riscos acima sem abandonar a ideia de partição de capacidade: **Máscara Dura + Pruning/Reset em Capacidade Fixa**. Em vez de crescer a rede, mantém-se a capacidade fixa e particiona-se: ordenar os pesos por `F_total` após A; os top-`K%` recebem máscara dura (congelados); os `(100-K)%` restantes sofrem reset/pruning (reinicializados) e ficam totalmente livres para B.

## Decisão (inferida da implementação da Fase 13, não de uma frase de síntese registrada)

A proposta do Antigravity — máscara dura + reset em capacidade fixa, sem crescimento estrutural — foi adotada quase literalmente na especificação da Fase 13 (Mecanismo C): granularidade por peso individual, importância via a mesma acumulação de Fisher do EWC, seleção por top-K% a cada *changepoint*, máscara cumulativa/monotônica, reset Xavier das posições livres. O crescimento estrutural foi descartado nesta rodada pelos riscos arquiteturais listados acima, e permanece descartado até a Fase 16 não ter resolvido o problema por outra via.
