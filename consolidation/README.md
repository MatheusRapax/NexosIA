# Fase 4: Sleep Consolidator & Generative Replay

Esta fase introduz um mecanismo inspirado biologicamente para mitigar o esquecimento catastrófico (Catastrophic Forgetting) sem utilizar buffers de memória de transições reais.

## Mecanismo de Sono Generativo e Rehearsal

O `SleepConsolidator` atua em três etapas fundamentais:
1. **Vigília (`observe`)**: Mantém médias móveis exaustivas (EMA) do estado de observação (`obs_mean` e `obs_std`) e da energia crônica da rede. Isso captura o domínio da tarefa atual sem armazenar as transições literais.
2. **Sono (`sleep_cycle`)**: Injeta ruído amostrando da distribuição empírica das observações e junta com uma ação válida sorteada, gerando pseudo-amostras ou "sonhos". A rede PCN processa esse sonho na direção normal `predict()` e "congela" a crença atual (o alvo). Esses pares formam a `dream_memory` de tamanho fixo, substituída a cada ciclo. Adicionalmente, aplica-se uma poda bayesiana (BMR) baseada na energia crônica.
3. **Rehearsal Intercalado (`rehearse`)**: Durante o treinamento de uma nova tarefa (Task B), o modelo revive as memórias sonhadas iterativamente ao final de cada época, blindando o conhecimento da Task A contra a sobreposição destrutiva causada pela Task B.

## Ausência de Buffer de Replay
Ao contrário do *Experience Replay* comum, `dream_memory` **NÃO VIOLA** a restrição biológica porque:
- É completamente auto-gerada pelo *World Model* e estatísticas condensadas.
- O vetor armazenado não corresponde a nenhuma transição real literal vivida pelo agente.
- Seu tamanho é fixo e estrito, e não acumula o histórico do agente.
