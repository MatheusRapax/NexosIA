# Fase 16 - Mecanismo D na Rede Escalada

## Parte 0: Sanity Check de Convergencia
- A_pos_A: 0.0361 (<= limite 0.0700) -> PASS
- B_pos_B: 0.0306 (<= limite 0.0593) -> PASS
*O sanity check passou de primeira, sem necessidade de ajuste de hiperparametros.*

## Tabelas de Resultados

### Baselines Escalados (4x)
| Modelo | A pos A | A pos B | B pos B | A pos C | B pos C |
|---|---|---|---|---|---|
| Naive | 0.0361 | 0.3704 | 0.0306 | 0.1930 | 0.0508 |
| Sleep | 0.0508 | 0.1083 | 0.1558 | 0.2554 | 0.2162 |

*(Foram disparados 4 changepoints em res_b_scaled)*

### Tabela de Calibracao (Mecanismo D)
| Gate | A pos A | A pos B | B pos B | A pos C | B pos C | Red A-apos-B | Red A-apos-C | Red B-apos-C |
|---|---|---|---|---|---|---|---|---|
| A(b) | 0.0508 | 0.1083 | 0.1558 | 0.2554 | 0.2162 | 70.76% | -32.34% | -325.44% |
| 0.25 | 0.1022 | 0.2000 | 0.3804 | 0.3412 | 0.5546 | 46.02% | -76.77% | -991.14% |
| 0.5 | 0.0485 | 0.1272 | 0.2689 | 0.1961 | 0.1735 | 65.67% | -1.60% | -241.37% |
| 0.75 | 0.0614 | 0.1491 | 0.1093 | 0.3343 | 0.1556 | 59.74% | -73.18% | -206.16% |

### Oraculo de Tarefa vs Gate Mais Recente
| Gate | A-pos-B (sem/com) | A-pos-C (sem/com) | B-pos-C (sem/com) |
|---|---|---|---|
| 0.25 | 0.2000 / 0.2000 | 0.3412 / 0.6661 | 0.5546 / 0.7181 |
| 0.5 | 0.1272 / 0.1272 | 0.1961 / 0.1961 | 0.1735 / 0.1735 |
| 0.75 | 0.1491 / 0.6701 | 0.3343 / 1.0240 | 0.1556 / 0.1556 |

## Conclusão
O Mecanismo D na rede escalada **não** satisfaz o criterio de aceite original (`B-pos-B < 0.35` E `RedA-C > 0`) para nenhum `gate_frac`. Embora `gate_frac=0.5` e `0.75` atinjam `B-pos-B < 0.35`, todos os valores de gate resultam em `Red A-apos-C < 0` (o esquecimento de A ao treinar C e pior do que na rede Naive baseline). Isso refuta a hipotese de deficit de capacidade como causa unica do fracasso do Mecanismo D (reabrindo a pergunta para a Rodada 11).


## Fase 17 - Correção de Sobreposição e Máscara Cumulativa

### Tabela de Calibração (Mecanismo D Corrigido)
| Gate | A pos A | A pos B | B pos B | A pos C | B pos C | Red A-apos-B | Red A-apos-C | Red B-apos-C |
|---|---|---|---|---|---|---|---|---|
| A(b) | 0.0508 | 0.1083 | 0.1558 | 0.2554 | 0.2162 | 70.76% | -32.34% | -325.44% |
| 0.25 | 0.1100 | 0.2298 | 0.3835 | 0.4965 | 0.7496 | 37.96% | -157.26% | -1374.76% |
| 0.5  | 0.0485 | 0.1272 | 0.2689 | 0.1961 | 0.1735 | 65.67% | -1.60% | -241.37% |
| 0.75 | 0.0614 | 0.1487 | 0.2302 | 0.2762 | 0.2527 | 59.85% | -43.13% | -397.26% |

### Logs de Changepoint e Fallbacks (Resumo)
- **Sleep baseline:** 4 changepoints (1 em A, 2 em B, 1 em C).
- **Gate 0.25:** 12 changepoints (6 em A, 0 em B, 6 em C). Fallback tier (b) disparou a partir do CP 5.
- **Gate 0.50:** 4 changepoints (2 em A, 0 em B, 2 em C). Fallback tier (b) disparou a partir do CP 3.
- **Gate 0.75:** 5 changepoints (1 em A, 3 em B, 1 em C). Fallback tier (b) no CP 2, e tier (c) (último recurso) acionado em quase todos (4 vezes).

### Conclusão
A correção da sobreposição global **não** resolveu a Red A-apos-C negativa. Nenhum dos `gate_frac` satisfez o critério original (`B-pos-B < 0.35` E `RedA-C > 0`), pois embora 0.5 e 0.75 preservem B razoavelmente, a métrica `RedA-C` permaneceu negativa (-1.6% e -43.13%).

O diagnóstico explica por que:
1. **Hipersensibilidade do detector (Gate 0.25):** Para `gate_frac=0.25`, o modelo gera um alto erro, causando 12 changepoints espúrios (vários no meio da região A e C). Essa quantidade massiva de CPs exaure a capacidade rapidamente, forçando fallbacks aleatórios já no CP 5.
2. **Exaustão prematura (Gate 0.5 e 0.75):** Para `gate_frac=0.5`, tivemos os 4 CPs esperados, mas como cada um exige 50% da rede, o tier (a) esgota já no CP 3 (exatamente ao iniciar a tarefa C), acionando fallback (b). Para `gate_frac=0.75`, a situação é pior: a capacidade esgota no CP 2, acionando o último recurso (tier c) sistematicamente, o que mistura as representações e causa esquecimento catastrófico.
