# Integration V4 - Múltiplas Regiões Sequenciais (A -> B -> C)

## Objetivo
Explorar o comportamento do Mecanismo A (Dual-Weight PCN + Priorized Replay Suavizado) validado na Fase 8 frente a *múltiplas transições* sequenciais. Na transição B -> C, o snapshot de importância (`visit_count`) é atualizado, apagando a memória do snapshot de A. Queríamos confirmar a previsão teórica de que a proteção da Região A se degradaria e o sistema protegeria preferencialmente a tarefa mais recente (B).

## Resultados Obtidos

### MSE Absoluto:
- MSE A pós A: 0.0583

#### Sem Sono (Baseline)
- MSE A pós B: 0.4256
- MSE B pós B: 0.0494
- MSE A pós C: 0.2078
- MSE B pós C: 0.0528

#### Com Sono (Mecanismo A Suavizado)
- MSE A pós B: 0.1437
- MSE B pós B: 0.3229
- MSE A pós C: 0.2436
- MSE B pós C: 0.0258

### Reduções Relativas de Erro
- **Redução em A após B:** 66.24% (consistente com o resultado de ~64.9% da Fase 8, validando a integridade do código).
- **Redução em A após C:** **-17.23%** (a primeira tarefa é esquecida / a proteção sobreviveu mal à segunda transição).
- **Redução em B após C:** **51.19%** (a tarefa intermediária foi fortemente protegida contra a terceira tarefa).

## Conclusão
**A previsão teórica foi CONFIRMADA na íntegra.** O uso de uma única *Slow Network* e um único *Snapshot de Importância* protege vigorosamente a tarefa imediatamente anterior (reduções de 66.24% e 51.19%), mas a proteção de tarefas passadas se "borra" ou satura assim que o snapshot é sobrescrito para proteger a transição atual. 

Esse limite arquitetônico documenta o ponto onde o Mecanismo A falha para o *Continual Learning* de longo prazo (>2 tarefas), exigindo abordagens mais escaláveis para proteger múltiplas memórias (ex: Mecanismo B - EWC local/Elastic Weight Consolidation com acúmulo de importância).
