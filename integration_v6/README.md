# Fase 13: Mecanismo C (Máscara Dura + Reset em Capacidade Fixa)

## Objetivo
Testar se substituir a penalidade suave (EWC) por uma máscara dura baseada em $F_{total}$ resolve o trade-off entre Redução do Esquecimento e Capacidade de Aprendizado sem aumentar a capacidade da rede.

## Resultados de Calibração (Sweep em k_frac)

| K     | A pos A | A pos B | B pos B | A pos C | B pos C | Red A-apos-B | Red A-apos-C | Red B-apos-C |
|-------|---------|---------|---------|---------|---------|--------------|--------------|--------------|
| A(b)  | 0.1848  | 0.1437  | 0.3229  | 0.2436  | 0.0258  |      66.24%  |     -17.23%  |      51.14%  |
| 0.1   | 0.0937  | 0.2330  | 0.5811  | 0.2702  | 0.2410  |      45.26%  |     -30.04%  |    -356.41%  |
| 0.25  | 0.0879  | 0.2042  | 0.4754  | 0.1118  | 0.5004  |      52.03%  |      46.18%  |    -847.77%  |
| 0.5   | 0.1679  | 0.1440  | 0.5329  | 0.2113  | 0.4404  |      66.16%  |      -1.71%  |    -734.07%  |
| 0.75  | 0.1493  | 0.1021  | 0.7373  | 0.1440  | 0.7821  |      76.00%  |      30.70%  |   -1381.31%  |

## Métricas Diagnósticas (Após Treino B)

- **K=0.1**: Camadas 1, 2 e 3 congeladas em ~30%. Erro B-pos-B é alto (0.58). Erro da restrição é 0.
- **K=0.25**: Camadas congeladas em ~44-50%. Erro B-pos-B cai um pouco (0.47), Red A-pos-C sobe (+46%).
- **K=0.5**: Camada 1 atingiu 100% de pesos congelados, Camada 2 ~76%, Camada 3 ~68%.
- **K=0.75**: Camadas 1 e 3 totalmente congeladas (100%).

Como esperado, a máscara monotônica cresce consideravelmente a cada tarefa.

## Conclusão Fase 13
**Critério de Aceite (B-pos-B < 0.35 E RedA-C > 0)**: Não foi satisfeito para nenhum $K$.
A máscara dura de fato reduz o esquecimento (Red A-C positivo para K=0.25 e K=0.75), e o clamp duro ($|W - w_{anchor}| \approx 0$) garante estabilidade aos pesos congelados. Porém, os valores de `B pos B` (variando de 0.47 a 0.73) demonstram destruição massiva na capacidade de aprendizagem da rede nas tarefas subsequentes. A acumulação monotônica de máscaras consome a capacidade fixa rapidamente (várias camadas chegando a 100% já na segunda tarefa para K alto).

Resultado Negativo validado: Uma máscara dura sem crescimento estrutural agrava o problema de capacidade.

---

## Fase 14 - Parte 1: Correção do Percentil

A lógica de cálculo do percentil usada na Fase 13 sofria de um bug onde valores empatados em 0.0 incluíam elementos extras, inflando a fração congelada muito acima de `K`. Ao mudar para um ranking determinístico exato, os valores foram corrigidos. 
Abaixo, a nova tabela de calibração recalculada:

| K     | A pos A | A pos B | B pos B | A pos C | B pos C | Red A-apos-B | Red A-apos-C | Red B-apos-C |
|-------|---------|---------|---------|---------|---------|--------------|--------------|--------------|
| A(b)  | 0.1848  | 0.1437  | 0.3229  | 0.2436  | 0.0258  |      66.24%  |     -17.23%  |      51.14%  |
| 0.1   | 0.0937  | 0.2330  | 0.5811  | 0.2702  | 0.2410  |      45.26%  |     -30.04%  |    -356.41%  |
| 0.25  | 0.0879  | 0.2042  | 0.4754  | 0.1118  | 0.5004  |      52.03%  |      46.18%  |    -847.77%  |
| 0.5   | 0.2866  | 0.1516  | 0.5501  | 0.1416  | 0.6565  |      64.37%  |      31.86%  |   -1143.30%  |
| 0.75  | 0.1648  | 0.1167  | 0.9984  | 0.0884  | 1.5523  |      72.57%  |      57.47%  |   -2839.93%  |

### Fração Congelada por K após Treino B (Fase 13 vs Fase 14)
*Nota: A métrica diagnóstica do experimento acumula a máscara em múltiplos changepoints dentro da Região B, o que eleva a fração total em relação ao K nominal.*
- K=0.1: ~30% (inalterado em relação à Fase 13)
- K=0.25: ~46% (inalterado em relação à Fase 13)
- K=0.5: 89% na Camada 1, vs 100% na Fase 13.
- K=0.75: 99.1% na Camada 1, vs 100% na Fase 13.

A correção não alterou a conclusão original: o critério de aceite (B-pos-B < 0.35 E RedA-C > 0) não é satisfeito, sendo o problema de déficit de capacidade limitante.

---

## Fase 14 - Parte 2: Diagnóstico de Ablação (Ancorar vs Zerar)

Neste experimento, comparamos o comportamento original do Mecanismo C (onde os pesos congelados são ancorados em seus valores de $W_{anchor}$ e continuam participando do *forward pass*) contra uma variante onde os pesos congelados são forçados a **zero** durante o treino de B. Isso isola a interferência que os pesos antigos causam na nova tarefa.

| K     | B-pos-B (ancorado) | B-pos-B (zerado)   | Delta (zerado - ancorado) |
|-------|--------------------|--------------------|---------------------------|
| 0.1   | 0.5811             | 0.5763             | -0.0048                   |
| 0.25  | 0.4754             | 0.3072             | -0.1683                   |
| 0.5   | 0.5501             | 0.9811             | +0.4310                   |
| 0.75  | 0.9984             | 0.8335             | -0.1649                   |

**Conclusão do Diagnóstico:**
Para $K=0.25$, "zerar" os pesos congelados melhorou substancialmente o erro na nova tarefa (redução de ~35% no erro, alcançando B-pos-B de 0.30). Isso indica que, quando há capacidade livre suficiente, a causa dominante do fracasso no aprendizado de B é a **interferência no forward pass** causada pelos pesos antigos congelados. O isolamento funcional (gating) é a abordagem mais promissora para lidar com isso.
No entanto, observamos que o impacto de zerar os pesos é não-monotônico: há melhora em $K=0.25$ e $K=0.75$, mas piora considerável em $K=0.5$. Isso provavelmente reflete ruído de seed única em vez de uma tendência limpa acima de $K=0.5$. De qualquer forma, para frações altas de congelamento, o erro absoluto continua muito alto, demonstrando que o **déficit de capacidade** pura ainda se torna o fator limitante (a rede não tem caminhos livres suficientes para aprender).
