# Debate — Rodada 6: Por que `mean→max` Não Bastou

> Sexta rodada de debate. A Fase 11 implementou a correção convergida na Rodada 5 (normalização saturante com `c_layer = mean(F_total)`) e produziu um retrocesso, não uma melhora: MSE B-pós-B entre 1.53 e 1.71, pior que tudo testado até então. Esta rodada diagnostica por quê e converge numa correção pontual (`mean→max`) que seria testada na Fase 12 — e também refutada, levando à Rodada 7. Ver [ARTIGO.md](ARTIGO.md) §11-§12 para os resultados experimentais.

## O resultado que motivou a rodada

Fase 11 (Anexo A da spec original): implementação fiel da síntese da Rodada 5 — `c_layer = mean(F_total da camada) + epsilon_c`, recalculado a cada `on_changepoint`, com o sweep de `lam` também alterado para `{0.1, 0.3, 0.5, 1.0}` (mais alto que o original da Fase 10, `{0.01,0.1,0.2,0.3}`). Resultado: MSE B-pós-B entre **1.53 e 1.71** em todos os `lam` — pior que o Mecanismo A isolado (0.3229) e pior que a própria Fase 10 sem normalização nenhuma.

## Diagnóstico inicial e sua refutação

Claude Code propôs uma hipótese de causa raiz: com `c_layer = mean(F_total)`, para os pesos de MAIOR `F_total` — exatamente os que a penalidade deveria proteger — o `ratio = F_total/(F_total+c_layer)` tende a 1. A saturação não reduz a penalidade nos pesos importantes, só redistribui a severidade; combinado com o sweep de `lam` subindo até 1.0 (vs. 0.3 antes), cada `apply_penalty` empurraria quase 100% desses pesos de volta à âncora.

O Antigravity concordou com a mecânica, mas discordou do enquadramento: **o EWC deve proteger pesos importantes com força — isso não é o defeito**. O erro real foi deixar `lam` subir a um valor absoluto de 1.0. Propôs duas correções simultâneas: reverter o sweep de `lam` para a faixa original (`0.01`–`0.3`) e trocar `c_layer = mean(F_total)` por `c_layer = max(F_total)` (ou um percentil alto, p99), garantindo que o ratio do peso mais importante da camada nunca ultrapasse 0.5 — eliminando o reset quase completo que a média permitia.

## Um segundo diagnóstico, levantado e depois retirado

Claude Code aprofundou com uma leitura direta do código (`integration_v5/run_integration_experiment.py`, linhas 107-109 e 141-143): `apply_penalty()` roda em TODO passo de treino sem nenhum gate — a variável `protecting` existe no código mas só controla o mecanismo de sono, nunca se o EWC está ativo. Isso levantou uma segunda hipótese: colapso geométrico da distância peso-âncora rumo a zero ao longo de milhares de passos, e uma proposta de gatear o EWC numa janela transitória de consolidação pós-changepoint.

O Antigravity refutou essa segunda hipótese com dois argumentos técnicos que se tornaram o ponto mais importante da rodada:

1. **O EWC é uma aproximação da posterior Bayesiana da tarefa anterior — deve ser permanente.** Desligá-lo numa janela causaria esquecimento catastrófico assim que a janela fechasse, contradizendo o propósito do próprio mecanismo.
2. **O colapso geométrico nunca chega a zero**, porque o gradiente da Tarefa B continua empurrando o peso a cada passo — no equilíbrio, a distância peso-âncora converge para `-(lr·grad_B)/(lam·ratio)`, não para zero. Com `lam=1.0` e `ratio≈1.0` (configuração real da Fase 11), essa distância de equilíbrio ficava em torno de apenas `0.05·grad_B` — um raio mínimo que travava o aprendizado de B, exatamente o sintoma observado.

Claude Code aceitou a refutação integralmente, retirando a proposta de janela transitória por quebrar a derivação Bayesiana do mecanismo. A contribuição que permaneceu, refinada: o colapso AMPLO da Fase 10/11 não era sobre duração da penalidade, era sobre COBERTURA — com `c_layer=mean`, aproximadamente metade dos pesos de cada camada já tinham `ratio>0.5`, dando raio de equilíbrio pequeno demais para boa parte da rede, não só para os pesos realmente críticos. Com `c_layer=max`, só o peso extremo da camada fica em `ratio=0.5`; o resto fica com `ratio` bem menor e raio de equilíbrio ordens de grandeza maior — liberando a maioria da rede para aprender B.

## Decisão — convergido (Fase 12)

1. Manter o EWC permanente — sem gating/janela de consolidação pós-changepoint (confirmado pela derivação Bayesiana).
2. Trocar `c_layer[l] = mean(F_total[l]) + epsilon_c` por `c_layer[l] = max(F_total[l]) + epsilon_c` em `on_changepoint`.
3. Reverter o sweep de `lam` para os valores originais da Fase 10: `{0.01, 0.1, 0.2, 0.3}`.
4. Adicionar uma métrica diagnóstica nova ao relatório: média de `|W - w_anchor|` por camada ao fim do treino de B, e fração de pesos com `ratio > 0.5` por camada — para verificar diretamente se a troca `mean→max` de fato abriu raio de liberdade para a maioria da rede, não só inferir isso indiretamente do MSE.

Esta síntese tornou-se a especificação da Fase 12. O resultado (ver ARTIGO.md §12) refutaria a própria hipótese desta rodada — a fração de `ratio>0.5` caiu a exatos 0%, confirmando o mecanismo técnico, mas o MSE de B-pós-B continuou falhando o critério de aceite em todos os `lam`, levando à Rodada 7 e à primeira revisão da premissa estrutural do Mecanismo B.
