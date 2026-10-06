# Debate — Rodada 9: O Executor Refuta uma Proposta do Arquiteto

> Nona rodada de debate. A Fase 14 corrigiu o bug do percentil, mediu evidência real de interferência de forward (K=0.25 melhorou ao zerar pesos congelados) e testou o Mecanismo D (Gating) — que perdeu severamente para o Mecanismo C, por déficit de capacidade da rede-base. Esta rodada decide os próximos passos e produz um dos achados metodológicos mais relevantes da sessão: o Executor, no papel de debatedor, refutou uma proposta do próprio Arquiteto antes de qualquer implementação. Ver [ARTIGO.md](ARTIGO.md) §15 e §19 (Trabalhos Futuros) para onde essa decisão levou.

## Ponto de abertura (Claude Code)

A Fase 14 deixou três pistas soltas. Claude Code propôs investigá-las em ordem de custo, do mais barato para o mais caro:

1. **Promover `HardMaskZeroTracker` (diagnóstico da Fase 14) a mecanismo real**, com sweep fino de `K` em torno de 0.15-0.35 — já que K=0.25 zerado chegou perto do critério de aceite (B-pós-B=0.3072) no diagnóstico original.
2. **Avaliação com oráculo de tarefa para o Gating** — guardar o gate histórico de cada região e usá-lo na avaliação, em vez do gate mais recente, para isolar quanto do fracasso da Fase 14 é artefato de avaliação vs. esquecimento real.
3. **Gating com capacidade escalada** — rede ~4x maior, condicional ao resultado dos itens 1-2.

## Contraponto (Antigravity): o Ponto 1 tem uma falha conceitual fatal

O Antigravity concordou com a lógica de priorização por custo, mas identificou que o Ponto 1 deveria ser descartado por completo, antes de qualquer implementação: o `HardMaskZeroTracker` zera exatamente os pesos de MAIOR `F_total` — os mais importantes para a Tarefa A. Zerá-los no forward destrói ativamente o conhecimento de A. Embora isso ajude B (removendo a interferência, o que explicava a melhora observada em B-pós-B), o resultado inevitável seria esquecimento catastrófico de A — uma métrica que o diagnóstico original da Fase 14 nunca chegou a medir, por ter sido concebido só para avaliar B. Zerar a principal informação de A anula o propósito de evitar o esquecimento; não deveria ser testado como solução.

O Ponto 2 (oráculo de tarefa) foi endossado como prioridade 1: barato e extremamente informativo, capaz de isolar se a falha do Gating era por falta de capacidade intrínseca ou só por usar o gate incorreto durante a avaliação de tarefas antigas. O Ponto 3 (escalar capacidade) foi reconhecido como o caminho natural caso o oráculo mostrasse que o erro permanecia inaceitável mesmo sabendo o gate certo.

## Síntese / decisão final (Claude Code)

Claude Code aceitou a correção integralmente, reconhecendo-a como um achado válido, não uma concessão de cortesia: o experimento proposto no Ponto 1 nunca havia medido o custo que o tornava inútil por definição. Decisão convergida: pular o Ponto 1 por completo; implementar o Ponto 2 (oráculo de tarefa para o Gating) como único entregável da Fase 15; condicionar o Ponto 3 ao resultado desse diagnóstico — se o oráculo mostrasse que a rede já isolava bem as tarefas, o problema passaria a ser inferência de tarefa; senão, a prioridade seguinte seria escalar a capacidade.

Este é um dos dois momentos da sessão (o outro sendo a Rodada 10, Sanity Check de Convergência) em que o papel de Debate do Executor refutou ou estendeu de forma substantiva uma proposta do Arquiteto antes da implementação — não apenas confirmou. O resultado da Fase 15 (oráculo mostrou um ganho real, mas parcial, em `gate_frac=0.25`) levou à Rodada 10, que escalou a rede diretamente (Fase 16).
