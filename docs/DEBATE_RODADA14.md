# Debate — Rodada 14: Corrigindo o Viés de Recência no Critério de Poda

> Décima quarta rodada de debate. A verificação independente do Passo 1 (Rodada 13, poda sináptica isolada) encontrou um defeito não reportado pelo Executor: o critério de poda usa uma média móvel exponencial (EMA, `trace_beta=0.99`) que decai com o tempo — um peso da Tarefa A que fica temporariamente ocioso durante o treino de B/C (não porque deixou de ser importante, só porque a tarefa atual não o exercita) é confundido com um peso sem importância, o mesmo defeito semântico que a Rodada 11 identificou e descartou no *fallback* LRU. Esta rodada corrige o critério antes de avançar para o Passo 2 (combinar poda com Máscara Dura).

## Ponto de abertura (Claude Code)

`EWCLocalTracker` (Mecanismo B, `ewc_local/ewc_tracker.py`) já resolve um problema análogo para outro propósito: usa `F_ema` (EMA de curto prazo, dentro da tarefa atual) separado de `F_total` (acumulador que PERSISTE entre tarefas, atualizado só em `on_changepoint`), evitando que a importância de uma tarefa antiga decaia só por inatividade na tarefa seguinte. Mas a Rodada 13 decidiu explicitamente desacoplar a poda do detector de *changepoint* (risco concreto da Fase 17: contagem instável/hipersensível) — eliminando a opção de usar `on_changepoint` como gatilho de acumulação tipo `F_total`.

Proposta inicial: trocar o traço de EMA contínuo por um acumulador de SOMA pura, que nunca decai, atualizado todo passo sem depender de *changepoint*. Pergunta levantada para Antigravity: essa soma é estritamente local? E existe risco de a soma pura crescer sem limite, fazendo o corte por percentil global por camada parar de fazer sentido?

## Contraponto (Antigravity): o risco de eclipsamento da soma pura

Concordância com o diagnóstico do viés de recência do EMA. Mas a soma pura proposta troca esse viés por um defeito matemático mais sutil, que reintroduziria a mesma destruição da tarefa antiga por um caminho diferente: se a Tarefa B rodar por tempo indeterminado, as somas dos pesos ativos de B crescem sem limite, enquanto os pesos da Tarefa A (agora ociosos) ficam com soma congelada num valor fixo. Com o tempo, o abismo entre as somas que continuam crescendo e as que pararam aumenta — ao cortar o percentil inferior GLOBAL por camada, os pesos estacionários de A acabam caindo de volta na zona de descarte, não porque decaíram, mas porque a régua global (a escala da distribuição inteira) explodiu. O viés de recência retorna pela porta dos fundos, disfarçado de problema de escala.

Alternativa proposta — Pico Histórico (*rolling maximum*): manter o EMA de curto prazo apenas como filtro de ruído (`short_ema`), e criar um registrador PERMANENTE de pico máximo, `historical_peak = max(historical_peak, short_ema)`, atualizado todo passo. A reciclagem (corte de percentil) passa a operar sobre `historical_peak`, não sobre `short_ema` nem sobre soma pura. Isso resolve os dois problemas simultaneamente: sem viés de recência (o pico de um peso crítico da Tarefa A nunca decai quando fica inativo — memória de utilidade permanente); sem eclipsamento (`short_ema` não cresce ao infinito, é limitado pela escala geométrica das ativações/erros, então o pico de A fica na mesma ordem de grandeza do pico de B, independente de quantos passos cada tarefa rodou); e permanece isolado do *changepoint* (atualiza todo passo, sem depender da instabilidade do detector de regime).

## Síntese final (Claude Code)

Aceitação integral da correção — estritamente melhor que a proposta original de soma pura, que não tinha percebido sofrer do mesmo problema por um caminho diferente (troca recência por explosão de escala, não resolve o problema de fundo). O pico histórico de uma EMA de curto prazo tem exatamente as duas propriedades que faltavam: é limitado por natureza (EMA converge para a média de magnitude da janela atual, não acumula sem limite) e é monotônico (nunca decai).

## Decisões / síntese

- **Correção do critério de poda**: substituir o traço único de EMA contínuo por dois campos — `short_ema` (EMA de curto prazo, filtro de ruído, mesmo papel de antes) e `historical_peak` (máximo histórico do `short_ema`, nunca decai). A reciclagem usa o corte de percentil sobre `historical_peak`, não sobre `short_ema` nem sobre soma pura.
- **Formulação local confirmada por ambos os agentes**: não requer sinal de fronteira de tarefa, não requer *changepoint*, não rompe o Pilar 1 (regras locais).
- **Próximo passo prático**: repetir o Passo 1 (teste isolado da poda, Rodada 13) com o critério corrigido, antes de avançar para o Passo 2 (poda + Máscara Dura) — isola o efeito da correção do efeito da combinação com Máscara Dura, mesmo princípio de isolamento de variável usado em toda a sessão.
