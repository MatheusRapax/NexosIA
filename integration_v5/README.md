# Fase 10 - EWC Local Online (Mecanismo B)

## Resultados da Calibração de `lam`

A suíte de testes rodou perfeitamente e varremos `lam` nos valores {0.01, 0.1, 0.2, 0.3, 1.0}.
O baseline do Mecanismo A (sem EWC) partiu de um MSE B pos B de 0.3229 e Redução A após C de -17.23%.

### Tabela de Calibração

| lam  | A pos A | A pos B | B pos B | A pos C | B pos C | Red A-apos-B | Red A-apos-C | Red B-apos-C |
|------|---------|---------|---------|---------|---------|--------------|--------------|--------------|
| A(b) | 0.1848  | 0.1437  | 0.3229  | 0.2436  | 0.0258  | 66.24%       | -17.23%      | 51.14%       |
| 0.01 | 0.1328  | 0.1100  | 0.4173  | 0.1915  | 0.1161  | 74.15%       | 7.82%        | -119.89%     |
| 0.10 | 0.2529  | 0.1310  | 0.8973  | 0.1079  | 1.1444  | 69.23%       | 48.05%       | -2067.50%    |
| 0.20 | 0.1419  | 0.1290  | 0.5554  | 0.1180  | 0.7573  | 69.68%       | 43.22%       | -1334.29%    |
| 0.30 | 0.1407  | 0.1171  | 0.7066  | 0.1144  | 0.6866  | 72.48%       | 44.93%       | -1200.36%    |
| 1.00 | 0.1021  | 0.0781  | 0.7402  | 0.0993  | 0.8064  | 81.65%       | 52.21%       | -1427.27%    |

## Análise do Trade-off e Conclusão

Nenhum dos lambdas testados atendeu simultaneamente aos critérios de sucesso (A-apos-C positivo E MSE B-pos-B < 0.10).
A rigidez no aprendizado de B decorre não apenas da penalidade EWC, mas do próprio Mecanismo A (que por si só já elevava o MSE B-pos-B para 0.3229, bem acima da meta de 0.10). Conforme o EWC é ativado, mesmo com um multiplicador brando como `lam=0.01`, a restrição de plasticidade em B sobe (0.4173). Ao aumentar `lam`, a retenção em A dispara positivamente (atingindo quase 50%), mas ao custo de arruinar por completo a plasticidade na tarefa B (erros > 0.70). 

**Nenhum valor serviu plenamente para equilibrar a equação.**
**Trabalhos futuros sugeridos:** Adotar uma modulação de `lam`. Em vez de um escalar global constante, o peso da penalidade deve depender de quanto `F_total` já acumulou naquele peso (ex: clipping), ou regular dinamicamente com base no erro preditivo, permitindo que a rede dobre a rigidez apenas onde absolutamente essencial, ou liberar os pesos menos informativos para aprender B sem restrições.

## Testes Adicionais
- O teste de equivalência bit-a-bit do PCN foi persistido com sucesso (`test_last_local_grad_equivalence.py`).
- O teste reduzido de integração (`test_mechanism_b.py`) foi implementado e atesta a ausência de crashes no fluxo do Mecanismo B de ponta a ponta.
- A regressão principal passou 100%.

**Status da Fase**: CONCLUIDO-COM-RESSALVA
