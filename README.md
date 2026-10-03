# Developmental PCN

Prova de conceito de uma arquitetura de inteligência artificial alternativa a LLMs: aprendizado local (sem backpropagation global), memória e raciocínio unificados no mesmo substrato de pesos plásticos, motivação intrínseca resistente a ruído, e consolidação sem buffer de experiências reais.

Nascida de um debate estruturado entre dois agentes de IA sobre como construir uma inteligência que cresça genuinamente por experiência, em vez de depender de escala bruta de hardware.

**Progresso ou parede já esperada? As duas coisas.** O limite de "task-free continual learning" (consolidar memória sem saber quando uma tarefa termina) foi previsto teoricamente antes de qualquer código, e os dois mecanismos testados até aqui (Fases 5 e 7) não o cruzaram por completo. Mas a *natureza* da resistência mudou entre gerações: a Fase 5 (mecanismo antigo) não produzia proteção mensurável independentemente do ajuste — uma parede estrutural. A Fase 7 (mecanismo novo, Dual-Weight PCN) produziu uma trajetória monotonicamente crescente a cada correção (2.84% → 17.79% → 25.09%, contra um limiar de 30%) — sinal de um mecanismo que funciona e está subdimensionado, não mal-adequado. Ver [docs/ARTIGO.md §8.1](docs/ARTIGO.md#81-progresso-ou-parede-já-esperada-as-duas-coisas-mas-não-da-mesma-forma) para a discussão completa.

## Comece aqui

- **[docs/ARTIGO.md](docs/ARTIGO.md)** — o relatório científico completo: motivação, arquitetura, metodologia, resultados de cada fase com números exatos, e os achados negativos das Fases 5 e 7.
- **[docs/DEBATE_ORIGINAL.md](docs/DEBATE_ORIGINAL.md)** — o debate (Rodada 1) que originou a arquitetura, com os 4 pilares e as limitações já previstas antes do código.
- **[docs/DEBATE_RODADA2.md](docs/DEBATE_RODADA2.md)** — segunda rodada de debate: diagnóstico do fracasso da Fase 5 e o mecanismo substituto (Dual-Weight PCN) que originou as Fases 6-7.
- **[docs/RELATORIO_TECNICO.md](docs/RELATORIO_TECNICO.md)** — processo de engenharia, bugs encontrados, como reproduzir cada experimento, e próximos passos concretos para quem for continuar.

## Estrutura

| Pasta | Fase | Status |
|---|---|---|
| `pcn_core/` | 1 — Motor de aprendizado local (PCN) | ✅ Aprovado |
| `active_inference/` | 2+3 — Ação unificada + motivação intrínseca | ✅ Aprovado |
| `consolidation/` | 4 — Consolidação via sono (oráculo de fronteira) | ✅ Aprovado |
| `integration/` | 5 — Agente único contínuo (SleepConsolidator) | ❌ Bloqueado (achado negativo) |
| `dual_weight/` | 6 — Detector de changepoint (Dual-Weight PCN) | ✅ Aprovado |
| `integration_v2/` | 7 — Reintegração com Dual-Weight PCN | ⚠️ Bloqueado (25.09% de 30% exigido — achado negativo quantificado, trajetória crescente) |

## Rodar tudo

```bash
pip install numpy torch scikit-learn pytest
pytest pcn_core/ active_inference/ consolidation/ dual_weight/ -q   # 15 testes, ~85s, deve passar 100%
pytest integration/ integration_v2/ -q                                # falhas esperadas, ver READMEs respectivos
```

Detalhes de cada script de demonstração em [docs/RELATORIO_TECNICO.md](docs/RELATORIO_TECNICO.md#3-como-reproduzir-cada-fase).
