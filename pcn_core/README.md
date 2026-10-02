# Predictive Coding Networks (PCN) vs Backpropagation

Este diretório implementa a Fase 1 da arquitetura *Developmental PCN*, demonstrando empiricamente que regras de atualização estritamente locais (*message passing*) conseguem atingir acurácia equivalente a um modelo com gradientes globais via *backpropagation*.

**Diferença conceitual principal:** No `BackpropMLP`, a rede exige duas passadas globais: uma passada *forward* até a saída final e uma passada *backward* (com a rede congelada) onde os gradientes são propagados em cadeia desde o fim até o início (exigindo que cada camada "lembre" e trave as computações passadas num grafo global unificado). Na `PredictiveCodingNetwork`, não existe passo *backward* global. Cada neurônio simplesmente envia uma predição (*mu*) para a camada acima, recebe um erro (*e*) dela, e ajusta sua própria atividade repetidas vezes até chegar a um consenso estático (fase de inferência local). Somente os erros das próprias vizinhanças imediatas ditam a mudança do peso sináptico. Essa distinção elimina a necessidade de `autograd` e acopla a aprendizagem naturalmente em substratos de *hardware* distribuído ou neuromórfico.

## Como rodar
```bash
python run_experiment.py
pytest tests/test_pcn_vs_backprop.py
```
