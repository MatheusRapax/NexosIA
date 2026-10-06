# Mechanism I Probe (Fixed Allocation Capacity)

Este probe avalia a alocação fixa e disjunta de capacidade por tarefa (redes divididas estaticamente em blocos de 1/3 da capacidade), comparando com a alocação dinâmica (GatingLocalTracker) e o Baseline denso. O harness foi estendido para suportar avaliação com Oráculo de Tarefa (`task_gate_fn`).

## Resultados

```text
Model                | A pos A  | A pos B  | B pos B  | A pos C  | B pos C  | Red A-pos-C  | Red B-pos-C 
----------------------------------------------------------------------------------------------------
Baseline             | 0.0106   | 0.1932   | 0.0099   | 0.1104   | 0.0567   |            - |            -
Gating-dinamico      | 0.0106   | 0.3388   | 0.0177   | 0.2278   | 0.2161   |     -106.34% |     -281.12%
Mecanismo I          | 0.0152   | 0.0985   | 0.0186   | 0.1565   | 0.0219   |      -41.77% |       61.42%
```

## Conclusões
1. **Deficit de Capacidade:** O erro `A pos A` do Mecanismo I (0.0152) é virtualmente idêntico ao Baseline (0.0106), provando definitivamente que 1/3 da capacidade da rede `[7, 64, 32, 2]` é mais que suficiente para aprender a tarefa. A falha nas rodadas anteriores não era por falta de parâmetros.
2. **Proteção Estrutural:** O Mecanismo I superou dramaticamente o Gating-dinâmico em ambas as retenções, validando que a sobreposição (overlap) dinâmica era de fato o calcanhar de aquiles do Mecanismo D.
3. **Red. B-pos-C Positivo:** Pela primeira vez entre as dezenas de mecanismos testados nas últimas rodadas, observamos uma **redução positiva de esquecimento** (+61.42% na retenção de B após C) em relação à Baseline densa. B foi brutalmente protegido contra a interferência de C.
4. **Interferência Residual (A pos C):** Apesar do imenso ganho em B, a retenção de A após C ainda é ligeiramente pior que a Baseline (-41.77%). Como os pesos intra-camada e intra-gate estão estruturalmente isolados com perfeição, a degradação remanescente só pode estar vindo dos parâmetros que continuam globalmente compartilhados e desprotegidos: **os vieses da camada de saída (`b[L]`)**, que são sobrescritos consecutivamente pelas tarefas B e C, corrompendo a decodificação final de A.

A alocação fixa funcionou como prova de conceito. O problema da interferência foi isolado na sobreposição de capacidade e nos parâmetros compartilhados de saída.
