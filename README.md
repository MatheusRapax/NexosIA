# Developmental PCN

Prova de conceito de uma arquitetura de inteligência artificial alternativa a LLMs: aprendizado local (sem backpropagation global), memória e raciocínio unificados no mesmo substrato de pesos plásticos, motivação intrínseca resistente a ruído, e consolidação sem buffer de experiências reais.

Nascida de um debate estruturado entre dois agentes de IA sobre como construir uma inteligência que cresça genuinamente por experiência, em vez de depender de escala bruta de hardware.

**Progresso através de uma parede que já era esperada.** O limite de "task-free continual learning" (consolidar memória sem saber quando uma tarefa termina) foi previsto teoricamente antes de qualquer código. Duas gerações de mecanismo (Fases 5 e 7) não o cruzaram — a primeira por estar estruturalmente mal-adequada (proteção ~0% independente do ajuste), a segunda por estar subdimensionada (2.84% → 17.79% → 25.09%, trajetória crescente mas parada antes do limiar de 30%). A terceira geração (Fase 8), nascida de uma observação do usuário sobre como o cérebro usa atenção/importância para não processar tudo igualmente, trocou "quanto repetir" por "o que importa repetir" — e cruzou o limiar com **64.91% de redução**. Ver [docs/ARTIGO.md §9.1](docs/ARTIGO.md#91-progresso-ou-parede-já-esperada-as-duas-coisas-mas-não-da-mesma-forma) e [§8](docs/ARTIGO.md#8-fase-8--replay-priorizado-o-mecanismo-que-cruzou-o-limiar) para a história completa.

## Comece aqui

- **[docs/ARTIGO.md](docs/ARTIGO.md)** — o relatório científico completo: motivação, arquitetura, metodologia, resultados de cada fase com números exatos, os dois achados negativos (Fases 5 e 7) e o mecanismo que finalmente funcionou (Fase 8).
- **[docs/DEBATE_ORIGINAL.md](docs/DEBATE_ORIGINAL.md)** — o debate (Rodada 1) que originou a arquitetura, com os 4 pilares e as limitações já previstas antes do código.
- **[docs/DEBATE_RODADA2.md](docs/DEBATE_RODADA2.md)** — segunda rodada: diagnóstico do fracasso da Fase 5 e o mecanismo substituto (Dual-Weight PCN) que originou as Fases 6-7.
- **[docs/DEBATE_RODADA3.md](docs/DEBATE_RODADA3.md)** — terceira rodada: a observação do usuário sobre atenção/importância, e como o mecanismo de replay priorizado destravou o projeto (Fase 8).
- **[docs/RELATORIO_TECNICO.md](docs/RELATORIO_TECNICO.md)** — processo de engenharia, bugs encontrados, como reproduzir cada experimento, e próximos passos concretos para quem for continuar.

## Estrutura

| Pasta | Fase | Status |
|---|---|---|
| `pcn_core/` | 1 — Motor de aprendizado local (PCN) | ✅ Aprovado |
| `active_inference/` | 2+3 — Ação unificada + motivação intrínseca | ✅ Aprovado |
| `consolidation/` | 4 — Consolidação via sono (oráculo de fronteira) | ✅ Aprovado |
| `integration/` | 5 — Agente único contínuo (SleepConsolidator) | ❌ Bloqueado histórico (substituído) |
| `dual_weight/` | 6 — Detector de changepoint (Dual-Weight PCN) | ✅ Aprovado |
| `integration_v2/` | 7 — Reintegração com Dual-Weight PCN (amostragem genérica) | ❌ Bloqueado histórico (25.09% de 30%, substituído) |
| `integration_v3/` | 8 — Reintegração com replay priorizado suavizado | ✅ **Aprovado — 64.91% de redução** |

## Rodar tudo

```bash
pip install numpy torch scikit-learn pytest
pytest pcn_core/ active_inference/ consolidation/ dual_weight/ -q   # 15 testes, ~85s, deve passar 100%
PYTHONPATH=".;./active_inference" pytest integration_v3/ -q           # Fase 8, 4 testes, ~131s, deve passar 100%
pytest integration/ integration_v2/ -q                                # historicos, falhas esperadas, ver READMEs respectivos
```

Detalhes de cada script de demonstração em [docs/RELATORIO_TECNICO.md](docs/RELATORIO_TECNICO.md#3-como-reproduzir-cada-fase).
