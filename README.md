# Developmental PCN

Prova de conceito de uma arquitetura de inteligência artificial alternativa a LLMs: aprendizado local (sem backpropagation global), memória e raciocínio unificados no mesmo substrato de pesos plásticos, motivação intrínseca resistente a ruído, e consolidação sem buffer de experiências reais.

Nascida de um debate estruturado entre dois agentes de IA sobre como construir uma inteligência que cresça genuinamente por experiência, em vez de depender de escala bruta de hardware.

## Comece aqui

- **[docs/ARTIGO.md](docs/ARTIGO.md)** — o relatório científico completo: motivação, arquitetura, metodologia, resultados de cada fase com números exatos, e o achado negativo da Fase 5.
- **[docs/DEBATE_ORIGINAL.md](docs/DEBATE_ORIGINAL.md)** — o debate que originou a arquitetura, com os 4 pilares e as limitações já previstas antes do código.
- **[docs/RELATORIO_TECNICO.md](docs/RELATORIO_TECNICO.md)** — processo de engenharia, bugs encontrados, como reproduzir cada experimento, e próximos passos concretos para quem for continuar.

## Estrutura

| Pasta | Fase | Status |
|---|---|---|
| `pcn_core/` | 1 — Motor de aprendizado local (PCN) | ✅ Aprovado |
| `active_inference/` | 2+3 — Ação unificada + motivação intrínseca | ✅ Aprovado |
| `consolidation/` | 4 — Consolidação via sono | ✅ Aprovado |
| `integration/` | 5 — Agente único contínuo | ❌ Bloqueado (achado negativo documentado) |

## Rodar tudo

```bash
pip install numpy torch scikit-learn pytest
pytest pcn_core/ active_inference/ consolidation/ -q   # 13 testes, ~100s, deve passar 100%
pytest integration/ -q                                   # falha esperada, ver integration/README.md
```

Detalhes de cada script de demonstração em [docs/RELATORIO_TECNICO.md](docs/RELATORIO_TECNICO.md#3-como-reproduzir-cada-fase).
