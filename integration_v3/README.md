# Integration V3 - Amostragem Priorizada Suavizada

## Mecanismo
Substituímos a amostragem de sonhos por gaussiana (`obs_mean`/`obs_std`) por uma amostragem proporcional à densidade real de visitas (`visit_count`) do `LearningProgressTracker`. 

Após detectarmos que a amostragem linear pura causava "loss of diversity" (onde a transição mais visitada concentrava mais de 95% das amostras e transições menos visitadas, incluindo a célula-objetivo, ficavam desprotegidas), implementamos **Priorização Suavizada** com os hiperparâmetros `alpha=0.5` e `epsilon=1.0`:
`adjusted = (visit_snapshot + epsilon) ** alpha`

Isso garantiu um piso mínimo de probabilidade para todas as transições válidas e mitigou a dominância do par mais visitado (caindo de ~96% para ~21.45%).

## Resultados
A condição sem sono reproduziu exatamente os números da Fase 7:
- MSE A pós A: 0.0583
- MSE A pós B (sem sono): 0.4404

Com o Mecanismo A (amostragem priorizada suavizada):
- MSE A pós B (com sono): 0.1545
- Redução relativa de erro: **64.91%**

## Comparação Explicita com a Fase 7
O resultado com a nova amostragem (64.91%) foi excepcionalmente SUPERIOR ao baseline com amostragem gaussiana da Fase 7 (25.09%) e também em comparação à versão com priorização linear pura (-11.06%).

Isto prova que uma amostragem estática proporcional de sonhos baseada no rastreamento real de visitas durante a Região A — desde que a diversidade seja mantida através de suavização apropriada — é fundamental para alcançar uma consolidação de memória eficiente e seletiva.
