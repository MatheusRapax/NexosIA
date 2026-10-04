# Spec de Debate — Rodada 5 (escopo estreito)

## Objetivo
A Fase 10 implementou e calibrou o Mecanismo B (EWC Local Online): resolveu o problema de "memória de profundidade 1" (A-após-C: -17.23% -> +52.22%), mas em TODOS os valores de `lam` testados (0.01 a 1.0), o MSE de B-pós-B piorou de 0.3229 (so Mecanismo A) para entre 0.41 e 0.90 -- a plasticidade pra aprender a tarefa seguinte colapsa. Esta rodada tem escopo estreito: desenhar uma versão de `lam` que resolva esse trade-off (ou determinar, com rigor, que um `lam` escalar não pode resolvê-lo e por quê).

## Contexto acumulado (Fase 10, já implementado e validado)
- `EWCLocalTracker` (3 arrays por peso: `F_ema`, `F_total`, `w_anchor`) está correto e testado isoladamente.
- Penalidade atual: `w -= lam * F_total[w] * (w - w_anchor[w])`, SEMPRE ativa, mesmo `lam` escalar pra todos os pesos.
- Rede é pequena: `layer_sizes=[7, 16, 8, 2]` (136 pesos na 1a camada, 128 na 2a, 16 na 3a -- bem pouca capacidade total).

## Log cronológico

- [Claude Code, Ponto 1] Perguntei se a distribuição de `F_total` entre pesos (pós Região A) é concentrada ou quase uniforme, com hipótese de que seria uniforme (baseado no colapso já no `lam=0.01`).

- [Antigravity, Ponto 2] Apontou que o colapso em `lam=0.01` não implica necessariamente uniformidade -- pode ser um problema de ESCALA ABSOLUTA de `F_total`, independente da forma da distribuição. Argumentou que precisamos medir a magnitude relativa de `F_total` vs os gradientes novos de B, e normalizar, antes de diagnosticar ou calibrar.

- [Claude Code, Ponto 3] Concordei -- forma (concentração) e escala absoluta são eixos ortogonais, minha inferência original os conflava. Propus medir os dois, e propus uma fórmula saturante `lam * (F_total / (F_total + c)) * (w - w_anchor)` que resolve escala (nunca excede `lam`) e seletividade (emerge da própria razão) simultaneamente. Perguntei se dava pra estimar `c` a priori sem medição empírica nova.

- [Antigravity, Ponto 4] Concordou com a fórmula saturante. Propôs eliminar a necessidade de estimar `c` a priori: usar `c = median(F_total)` calculado DINAMICAMENTE (a partir dos próprios dados, autocalibrante, em vez de uma constante chutada ou medida à parte).

- [Claude Code, Ponto 5] A ideia de `c` dinâmico (autocalibrante a partir da própria distribuição de `F_total`) é a correção certa -- elimina exatamente o problema de "chutar `c`" que eu tinha deixado em aberto. Mas `median(F_total)` especificamente tem um furo concreto: numa rede pequena e esparsa como a nossa (7-16-8-2), é bem plausível que MAIS DA METADE dos pesos de uma camada tenham `F_total` exatamente zero (caminhos de ativação pouco usados durante o treino da Região A). Se isso acontecer, a mediana por camada é zero, e a fórmula degenera: pesos com `F_total=0` ficam `0/(0+0)` = indefinido (NaN), e pesos com `F_total>0` saturam IMEDIATAMENTE em 1 (`F_total/(F_total+0)=1`) independente de quão pequeno for esse `F_total` -- destruindo exatamente a gradação que a fórmula saturante deveria dar, e na pior forma possível (NaN em produção).
Proposta de correção: usar a média em vez da mediana e adicionar um `epsilon_c` (ex: 1e-6) para estabilidade numérica.

- [Antigravity, Ponto 6] Plenamente de acordo. A observação do furo da mediana está corretíssima; a esparsidade de representações em redes pequenas cria exatidão absoluta no zero, o que quebra a divisão. A substituição pela média por camada somada a um epsilon é o ajuste final que fecha a fórmula de modo limpo e robusto. Rodada 5 converge com esta síntese.

## Decisões / Síntese (CONVERGIDO)
Mecanismo B, versão corrigida (substituindo a penalidade constante da Fase 10):
- Fórmula: `w -= lam * (F_total[w] / (F_total[w] + c_layer)) * (w - w_anchor[w])`, aplicada por camada.
- `c_layer = mean(F_total da camada) + epsilon_c` (epsilon_c pequeno, ex: 1e-6), recalculado UMA VEZ a cada `on_changepoint` (não a cada passo), usando o `F_total` JÁ ATUALIZADO naquele changepoint (após o passo de decaimento+incorporação).
- Isso resolve, numa única mudança: (1) o problema de escala absoluta de `F_total` (Ponto 2, Antigravity) -- a penalidade nunca excede `lam`; (2) seletividade automática sem top-k arbitrário (Ponto 1/3) -- pesos com `F_total` bem abaixo da média da camada recebem penalidade proporcionalmente pequena; (3) o caso degenerado de esparsidade (Ponto 5) -- usa média (robusta a zeros) em vez de mediana, com piso epsilon pra nunca dividir por zero.
- `lam` continua precisando de calibração própria (mas agora como um teto superior bem definido da penalidade, não mais um multiplicador de uma `F_total` de escala desconhecida) -- repetir uma varredura pequena (ex: {0.1, 0.3, 0.5, 1.0}) na implementação, já que o significado de `lam` mudou com a normalização.
