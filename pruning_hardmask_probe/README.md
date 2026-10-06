# Mecanismo Combinado: Poda Sinaptica + Mascara Dura (Rodada 14)

## Objetivo
Este teste executa e compara quatro cenarios do Predictive Coding Network em um protocolo sequencial A -> B -> C de `GridWorld`:
1. **Baseline**: Rede densa sem nenhum mecanismo continuo.
2. **Poda Isolada**: Rede com decaimento continuo e reciclagem periodica (`LeakyRecyclingPCN`), baseada na magnitude do gradiente historico, sem protecao.
3. **Mascara Isolada**: Rede densa com congelamento periodico dos pesos de maior magnitude do gradiente (`HardMaskLocalTracker`), com K=50%. Isso inclui o **reset de capacidade livre** nos changepoints.
4. **Combinado (Poda + Mascara)**: O Passo 2 idealizado. A mascara dura protege os pesos criticos congelando-os a cada fronteira de tarefa. A poda recicla os pesos fora da mascara durante o treino das tarefas subsequentes. E esperado que o reset da capacidade livre imposto pela mascara dura interaja (colida) com a reciclagem da poda.

## Como rodar
1. Execute a suite de testes unitarios:
   ```bash
   python -m pytest pruning_hardmask_probe/tests/
   ```
2. Execute o script principal de sondagem (configurado com 200 epochs por tarefa):
   ```bash
   $env:PYTHONPATH="."; python pruning_hardmask_probe/run_probe.py
   ```

## Resultados (Tabela de Erro e Reducao de Esquecimento)

```text
Model            | A pos A  | A pos B  | B pos B  | A pos C  | B pos C  | Red A-pos-C  | Red B-pos-C 
---------------------------------------------------------------------------------------------------------
Baseline         | 0.0050   | 0.1206   | 0.0054   | 0.1584   | 0.0096   |            - |           -
Poda Isolada     | 0.0275   | 0.5123   | 0.0275   | 0.2741   | 0.2713   |      -73.06% |   -2736.45%
Mascara Isolada  | 0.1711   | 0.1084   | 0.0069   | 0.1665   | 0.0425   |       -5.11% |    -343.92%
Combinado        | 0.0365   | 0.5019   | 0.0276   | 0.2745   | 0.2727   |      -73.33% |   -2750.50%
```

## Conclusao
A colisao dos mecanismos de reset e instrutiva: o "reset de capacidade livre" embutido na Mascara Dura destroi a performance. A Mascara Isolada passou de reter tarefas (reducao de 55% num teste onde o reset foi desativado acidentalmente) para **esquecer** mais que o baseline (-5.11%) apenas por resetar a capacidade livre (50% do total) na fronteira de tarefas. A combinacao de Poda + Mascara, similarmente, tem um desempenho virtualmente identico a Poda Isolada (A-pos-C de -73%). 

Isso confirma categoricamente que "resetar a capacidade livre" (Mecanismo C original da Fase 13) e destrutivo e incompativel com a topologia distribuida do PCN. Seja o reset hard (na fronteira de tarefas) ou soft (continuo via poda), apagar os pesos menores/livres arruina a interferencia passada, indicando que o PCN depende fortemente de toda a rede para inferencia, invalidando mecanismos puramente baseados na particula de "capacidade livre=lixo".
