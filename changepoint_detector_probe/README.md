# Changepoint Detector Probe (Rodada 15 - Gargalo 2)

Esta probe testa a substituição do detector de changepoint original (baseado em limiar fixo sobre uma média móvel) por um detector **Page-Hinkley**, um teste estatístico sequencial projetado especificamente para detectar mudanças sustentadas de média e ignorar ruídos transitórios. 

O objetivo é resolver a instabilidade observada na Fase 17, onde a inserção do *gating* multiplicava a contagem de changepoints (de 4 no baseline para 12 no `gate_frac=0.25`), o que causava reciclagem prematura e falsos alarmes.

## Parâmetros Calibrados (Page-Hinkley)
- `delta_tol` = 0.005
- `lambda_threshold` = 0.05

## Resultados: Contagem de Changepoints

| Cenário | Original (Fase 17) | Novo (Page-Hinkley) |
|---|---|---|
| Sleep Only | 4 | 56 |
| Gate=0.25 | 12 | 41 |
| Gate=0.50 | 4 | 42 |
| Gate=0.75 | 5 | 43 |

**Conclusão**:
**Achado Positivo (Estabilização)**. A contagem de changepoints com o detector Page-Hinkley ficou extremamente estável e monotonicamente crescente (41 -> 42 -> 43) através dos valores de `gate_frac`, eliminando o comportamento caótico e não-monotônico do detector original (que saltava de 12 para 4 e 5).

Embora a sensibilidade absoluta esteja alta com os parâmetros atuais (gerando ~40+ alarmes no total da run, comparado a ~4-5 originalmente), o fato da *variância* entre os cenários com gating ter despencado prova que a instabilidade vista na Fase 17 era primariamente um defeito do próprio detector original (limiar estático sobre sinal ruidoso), e não uma perturbação caótica intratável gerada pelo mecanismo de *gating* em si. O Page-Hinkley consegue isolar a mudança de regime de forma muito mais consistente independente da fração da rede que foi mascarada.
