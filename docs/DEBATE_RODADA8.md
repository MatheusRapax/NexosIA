# Debate — Rodada 8: Separando Interferência de Forward de Déficit de Capacidade

> Oitava rodada de debate, a primeira conduzida integralmente nesta sessão de trabalho, com o Claude Code assumindo temporariamente o papel de debatedor (não só de Arquiteto). A Fase 13 (Mecanismo C, máscara dura) refutou a hipótese da Rodada 7: nenhum `K` satisfez o critério de aceite, e um bug de ferramenta (empate no corte por percentil) foi descoberto no processo. Esta rodada decide a ordem de investigação para a Fase 14 e produz, como subproduto, o Mecanismo D (Context-dependent Gating). Ver [ARTIGO.md](ARTIGO.md) §14 para os resultados experimentais.

## Ponto de abertura (Claude Code)

Antes de escolher entre as três pistas deixadas pela Fase 13 — (a) orçamento de capacidade fixo, (b) resgatar crescimento estrutural, (c) corrigir o bug do percentil —, Claude Code propôs isolar qual fração do colapso de B-pós-B vinha de duas causas distintas, até então tratadas como uma só:

1. **Interferência residual no forward**: os pesos congelados/ancorados de A continuam ativos no forward de B (tanto no EWC quanto na máscara dura). B pode estar gastando capacidade livre só para cancelar esse sinal residual, não para aprender sua própria função.
2. **Déficit puro de capacidade livre**: mesmo sem nenhuma interferência, pode não sobrar unidades/pesos livres suficientes para representar a função de B.

As duas causas pedem soluções diferentes — isolamento funcional (crescimento, colunas separadas, ou zerar em vez de ancorar) vs. mais capacidade livre (orçamento melhor, rede maior). Proposta de experimento barato, reaproveitando o código da Fase 13: uma variante diagnóstica do `HardMaskLocalTracker` que zera (em vez de ancorar) os pesos congelados durante o treino de B, comparando o B-pós-B resultante contra a versão ancorada para o mesmo `K`.

## Contraponto e extensão (Antigravity)

O Antigravity concordou com a distinção e com o valor do experimento de ablação, mas levantou uma ressalva de ordem crítica: o bug de empate do percentil (achado diagnóstico da própria Fase 13) precisava ser corrigido ANTES do experimento de ablação — caso contrário, o `K` nominal usado no diagnóstico não corresponderia ao `K` real aplicado, e o experimento poderia superestimar o déficit de capacidade.

Como extensão, propôs uma alternativa ao crescimento estrutural (já descartado na Rodada 7 pelos riscos arquiteturais do PCN): **Context-dependent Gating** (Masse, Grant & Freedman, *PNAS* 2018 / XdG), batizado de Mecanismo D. Em vez de congelar/ancorar PESOS, sortear uma máscara de gate binária por UNIDADE OCULTA a cada *changepoint*, aplicada multiplicativamente na ativação durante o forward (e portanto no gradiente, via a regra da cadeia). Isso ataca as duas causas ao mesmo tempo, por construção: unidades desligadas contribuem exatamente zero no forward (sem interferência residual, sem precisar ancorar nada), e os gates não são permanentes — podem se reaproveitar entre tarefas, evitando o esgotamento monotônico de capacidade da máscara dura. Não exige nenhuma mudança estrutural no PCN.

## Síntese / decisão final (Claude Code)

Claude Code concordou com a ordem proposta pelo Antigravity e reconheceu a proposta de Gating como superior à sua própria ideia inicial — ataca interferência de forward e esgotamento de capacidade simultaneamente, sem os riscos arquiteturais do crescimento. Decisão convergida para a Fase 14, em três entregáveis sequenciais:

1. Corrigir o `HardMaskLocalTracker`: substituir o corte por `np.percentile` (que empata em valores repetidos de `F_total`) por ranking determinístico (`argsort` estável + `ceil(k_frac·N)` elementos exatos).
2. Com o `K` corrigido, rodar o diagnóstico de ablação (zerar vs. ancorar) para medir o peso relativo de cada causa.
3. Implementar Context-dependent Gating (Mecanismo D) como protótipo paralelo, testado na mesma condição A/B/C com os mesmos baselines, para comparação direta com o Mecanismo C.

Crescimento estrutural permanece descartado. Questões deixadas explicitamente em aberto para quando a Fase 14 fosse especificada: fração de overlap entre gates de tarefas diferentes, e se overlaps residuais precisariam de proteção adicional — nenhuma delas bloqueava o início do trabalho.
