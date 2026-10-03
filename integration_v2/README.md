# Integração V2: Dual-Weight PCN e Active Inference (BLOQUEADO)

## Mecanismo de Integração

O experimento tentou utilizar a **Fast Network** do `DualWeightPCN` como o modelo de mundo do `ActiveInferenceAgent`. A logica projetada seria:
- O sinal `smoothed_delta` rastreia a diferença de erro entre a rede Slow (ancorada no passado) e a Fast.
- Em bordas de subida (`smoothed_delta > threshold`), o sistema entra em modo permanente de `protecting = True`.
- Nesse modo, a cada passo real de exploração são realizados também 4 passos de sonho (ensaio): as observações são amostradas da média e desvio padrão históricos (`obs_mean`, `obs_std`), a Slow Network atua como professora (`y_dream`), e a Fast Network é ensinada a não esquecer.

## Diagnóstico do Bloqueio

O design atual falhou iterativamente em bater o critério mínimo de 30% de redução no erro, resultando em um **BLOQUEIO DEFINITIVO (Achado Negativo Legítimo)**. A trajetória empírica registrou a seguinte progressão de retenção de memória frente a 6000 passos do agente:
- Burst Único no Trigger (original): 2.84%
- Ensaio Contínuo Permanente (ratio 1:1): 17.79%
- Ensaio Contínuo Majoritário (ratio 4:1): **25.09%**

Esses dados isolaram a limitação central do acoplamento: embora amplificar e tornar constante a injeção dos sonhos tenha mitigado o esquecimento em 1/4 do total do erro, a Slow Network sozinha e os rastreamentos estatísticos baseados em EMA são gradualmente corrompidos num horizonte de milhares de passos do Active Inference. Não é viável escalar a proporção indefinidamente, e uma nova arquitetura (possivelmente episódica) seria necessária.

### Parâmetros Experimentados (Finais)
- `threshold = 0.015`
- `debounce = 100` (passos)
- `beta_slow = 0.00005`
- `ratio = 4 ensaios por passo real`
