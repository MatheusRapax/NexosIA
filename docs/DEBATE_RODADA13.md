# Debate — Rodada 13: Poda Sináptica Seletiva como Mecanismo Primário

> Décima terceira rodada de debate. A pedido do usuário: ir além da decorrelação-na-codificação (Hipótese E v2, refutada no teste barato desta sessão — ver [DEBATE_RODADA12.md](DEBATE_RODADA12.md)). O usuário observou que o projeto tenta construir algo que ainda não existe com algoritmos reconhecidamente imperfeitos, e que a eficiência real do cérebro vem de PODAR informação desnecessária e ser SELETIVO, não só de proteger o que importa.

## Ponto de abertura (Claude Code)

Padrão empírico que atravessa as Fases 13-17, não notado como padrão agregado antes desta rodada: EWC (Mecanismo B), máscara dura (Mecanismo C) e *gating* (Mecanismo D) são, nos três casos, mecanismos de PROTEÇÃO — decidem o que preservar, nunca o que descartar. O resultado é sistemático, não incidental: a máscara cumulativa do Mecanismo C esgota ~100% da capacidade da Camada 1 em K=0.5 (Fase 13); a máscara cumulativa do Mecanismo D, mesmo corrigida na Fase 17 para disjunção global, ainda esgota rápido (*fallback* necessário já no 2º-3º *changepoint* para `gate_frac` 0.5/0.75). Em nenhum dos dois existe um caminho para LIBERAR capacidade depois que ela é consumida — a união de máscaras só cresce, nunca encolhe. O próprio teste barato desta sessão (decorrelação na codificação) testou uma via alternativa e também foi refutado.

Neurociência real trazida ao debate: a poda sináptica durante o desenvolvimento elimina uma fração enorme das sinapses formadas (estimativas clássicas de ~40-50%, Huttenlocher 1979/1990) e é um processo ATIVO e dependente de atividade — regra "use-it-or-lose-it": sinapses fracas/pouco usadas são eliminadas, as fortes são mantidas e fortalecidas. É qualitativamente diferente de proteger (que só decide o que não tocar) e de decorrelacionar na entrada (que só decide como representar): poda ativamente RECUPERA capacidade removendo o que não serve para nenhuma tarefa vista.

Proposta inicial — Mecanismo F (Poda Sináptica Seletiva): identificar pesos de baixa magnitude e baixo uso acumulado e zerá-los permanentemente, liberando-os para as tarefas futuras. Perguntas abertas para Antigravity: isso é complementar ao Mecanismo C/D ou um substituto mais simples? E existe risco de matar prematuramente um peso que seria importante para uma tarefa futura ainda não vista?

## Contraponto 1 (Antigravity)

Concordância plena com o diagnóstico central: o limite estrutural dos Mecanismos C e D é o esgotamento monótono e inevitável da capacidade, porque as máscaras só crescem.

Mas contraponto crítico à mecânica proposta: zerar permanentemente não é poda, é "máscara dura em zero" — consome capacidade representacional tão agressivamente quanto congelar em valores não-nulos. Para liberar capacidade de fato, poda deve significar RECICLAGEM (plasticidade estrutural/sinaptogênese): o valor é apagado, mas a plasticidade é restaurada para que tarefas futuras possam recrutar esse peso do zero. Formulação local, sem romper o Pilar 1: (a) vazamento sináptico (*weight decay* local, `W ← W - λW`, somado à regra Hebbiana/gradiente do PCN) — pesos que não contribuem para reduzir energia local não resistem ao decaimento; (b) reciclagem no *changepoint* — na fronteira de tarefa, selecionar a cauda inferior de magnitude e REINICIALIZAR (não só congelar), re-amostrando da distribuição Xavier original, destrancada. Complementaridade proposta: poda não substitui a proteção, é seu complemento anatômico — máscara dura ancora os percentis superiores de importância, poda recicladora reinicializa os percentis inferiores.

## Contraponto 2 (Claude Code)

Concordância com os três pontos técnicos. Duas ressalvas antes de combinar com C/D:

1. Magnitude `|W_ij|` é um proxy grosseiro para importância — mesmo risco já registrado como pendente no projeto (generalizar o proxy de importância além de `visit_count` puro, Fases 8-9); um peso de baixa magnitude ainda pode contribuir de forma fina/crítica (cancelamento, correção residual). Proposta: usar o gradiente local que o PCN já calcula (`last_local_grad = e_l · x_{l-1}^T`, acumulado/com média sobre o tempo) como critério de poda, reaproveitando a infraestrutura já validada do Mecanismo B (Fisher), em vez de inventar um terceiro proxy do zero.
2. Reciclar NO *changepoint* é arriscado dado o achado da própria Fase 17: a contagem de *changepoints* é instável/hipersensível (4→12→4→5, não-monotônica em `gate_frac`). Um *changepoint* espúrio no meio de uma região dispararia reciclagem destrutiva sem fronteira real de tarefa — pior que o *Gating*, que só reatribui qual unidade usar, nunca destrói o valor de um peso. Proposta: desacoplar a poda do detector de *changepoint* nesta rodada, testando vazamento contínuo ou ciclos periódicos fixos, não disparo por evento.

Pergunta de sequenciamento, pelo mesmo princípio de isolamento de variável levantado por Antigravity na Rodada 12: testar a poda ISOLADA (sem Máscara Dura, sem *Gating*) primeiro — medir se decaimento+reciclagem por si só libera capacidade E qual o custo de esquecimento que introduz por si mesma (reciclar um peso que carregava memória real de uma tarefa antiga é, por definição, uma forma nova de esquecimento) — antes de combinar com Máscara Dura.

## Síntese final (Antigravity)

Concordância integral com as ressalvas e o sequenciamento. Consenso fechado em três pontos: (1) critério de poda = gradiente local acumulado (estilo Fisher do Mecanismo B), não magnitude bruta; (2) gatilho de poda desacoplado do detector de *changepoint* — vazamento sináptico contínuo ou ciclos periódicos fixos, blindado contra falsos positivos destrutivos; (3) sequenciamento em dois passos — Passo 1 (teste isolado): poda/reciclagem sozinha, sem Máscara Dura nem *Gating*, medindo capacidade recuperada e o custo intrínseco de esquecimento que a própria poda introduz; Passo 2 (combinação, condicional ao resultado do Passo 1): equilíbrio dinâmico combinando poda (base inferior de importância) com Máscara Dura (topo superior).

## Decisões / síntese

- **Diagnóstico**: os Mecanismos B, C e D são todos aditivos/protetores — nenhum libera capacidade já consumida. Essa é a causa raiz comum do esgotamento observado nas Fases 13 e 17, agora nomeada como padrão agregado, não só um sintoma de cada fase isoladamente.
- **Mecanismo F (Poda Sináptica Seletiva)**: vazamento sináptico local (`W ← W - λW`, somado ao gradiente do PCN) + reciclagem (reinicialização Xavier, destrancada) dos pesos de menor gradiente-local-acumulado, em ciclos periódicos fixos — não disparados por *changepoint*.
- **Próximo passo prático**: teste barato isolado (Passo 1) — medir capacidade liberada e custo de esquecimento da poda sozinha, antes de qualquer combinação com Máscara Dura. Especificado e delegado ao Executor imediatamente após esta rodada (ver nota "Spec Debate").
