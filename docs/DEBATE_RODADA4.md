# Debate — Rodada 4: Projetando o Mecanismo B (EWC Local Online)

> Quarta rodada de debate, convocada para resolver o limite de "memória de profundidade 1" que a Fase 9 acabou de medir no Mecanismo A (replay priorizado): ele protege fortemente a tarefa imediatamente anterior, mas perde a proteção da tarefa mais antiga a cada nova transição (A-após-B: 66.24%; A-após-C: -17.23%; B-após-C: 51.19%). O Mecanismo B deveria ser um COMPLEMENTO ao Mecanismo A, não um substituto. Ver [ARTIGO.md](ARTIGO.md) §10 para os resultados experimentais (Fase 10) que testaram esta proposta.

## Contexto herdado da Rodada 3

Duas correções já haviam sido registradas na Rodada 3 como trabalho futuro, mas nunca especificadas em detalhe: (1) o proxy correto de importância por peso é a Fisher empírica local, `(e_l[i] · x_{l-1}[j])²` — já implicitamente calculada pela regra de aprendizado local existente, não "erro acumulado" (que penalizaria pesos errados, não importantes); (2) proteger um peso só reduzindo sua taxa de aprendizado sofreria do mesmo drift lento-mas-eventual que já havia quebrado o `beta_slow` isolado em horizontes longos — proteção real exige uma âncora (um valor de referência do peso capturado no momento em que ele era importante), custando um terceiro array de estado por peso, além de fast e slow.

## Questão 1 — Acúmulo de Fisher ao longo de 3+ tarefas

Claude Code abriu com a hipótese central: diferente do snapshot único de `visit_count` do Mecanismo A (SUBSTITUÍDO a cada changepoint, por isso tem profundidade 1), a Fisher por peso poderia ser ACUMULADA por soma a cada changepoint (`F_total += F_tarefa_atual`), nunca descartando tarefas anteriores — escapando estruturalmente da profundidade 1 porque soma preserva histórico enquanto substituição de snapshot não. Risco levantado desde o início: Fisher acumulada sem decaimento cresceria sem limite e acabaria congelando a rede inteira — o problema oposto, mas igualmente real.

Em vez de reinventar o acúmulo e a âncora do zero, a proposta foi adaptar diretamente a formulação de **EWC Online multi-tarefa** (Kirkpatrick et al., 2017; Progress & Compress, Schwarz et al., 2018) ao proxy de Fisher local já disponível, em vez de usar a Fisher diagonal via backprop global da formulação original.

## Questão 2 — Desenho exato da âncora, e o contraponto decisivo do Antigravity

A proposta inicial de âncora (`w_anchor` = valor do peso no momento do changepoint, com penalidade `λ·F_total·(w - w_anchor)²`) deixava uma pergunta em aberto: a âncora deveria ser atualizada a cada changepoint (como o `visit_count` hoje) ou fixada uma única vez? Atualizá-la sempre pareceria reintroduzir profundidade 1 pela porta dos fundos; nunca atualizá-la ignoraria que o peso ótimo para A pode não ser o ótimo para A+B+C.

O Antigravity resolveu isso com dois contrapontos que se tornaram o núcleo técnico do mecanismo:

1. **A Fisher não pode ser acumulada durante a tarefa inteira.** No EWC original, a Fisher é calculada com a rede já convergida. Acumular o proxy `(e_l·x)²` durante toda a tarefa deixaria os erros transientes do início dominarem o valor — medindo "o que mudou muito" em vez de "o que é importante manter estável". Solução: `F_tarefa_atual` deve ser um snapshot de uma EMA rápida do proxy, tirado no exato momento do changepoint, não uma soma bruta ao longo do tempo.
2. **A âncora deve ser uma média ponderada por Fisher, com decaimento aplicado ANTES da ponderação** — não um valor substituído nem uma média simples:
   - `F_old_decayed = gamma · F_total`
   - `w_anchor_novo = (F_old_decayed · w_anchor_antigo + F_tarefa_atual · w_atual) / (F_old_decayed + F_tarefa_atual)`
   - `F_total_novo = F_old_decayed + F_tarefa_atual`

Isso restaura plasticidade de forma suave (informação antiga perde peso gradualmente) e resolve a profundidade 1 estruturalmente: pesos seguem acumulando Fisher de múltiplas tarefas, nunca "esquecendo" todas as anteriores de uma vez como o snapshot de `visit_count`.

## Fechamento: três restrições de implementação

Antes de declarar convergência, Claude Code fechou três pontos que a formulação acima ainda não deixava explícitos:

- **Custo de estado**: 3 arrays novos por peso (`F_ema`, `F_total`, `w_anchor`), além de fast/slow já existentes. `F_ema` reusa o mesmo `beta_delta=0.1` já validado para o `smoothed_delta` do detector de changepoint, em vez de uma nova constante de tempo.
- **Localidade preservada**: a penalidade `-λ·F_total[w]·(w - w_anchor[w])`, somada ao update local existente, usa só o próprio estado daquele peso — nenhuma dependência cross-peso é introduzida. Confirmado explicitamente por ser a restrição arquitetural mais importante do projeto.
- **Penalidade sempre ativa**: diferente do Mecanismo A (bursty, só durante a janela de "protecting"), a penalidade EWC roda em todo passo de treino, real ou sonho — comportamento esperado de regularização contínua. O trade-off plasticidade-vs-estabilidade já medido no Mecanismo A (B-pós-B pior com sono ativo: 0.3229 vs. 0.0494, Fase 9) foi antecipado aqui como esperado, não tratado como surpresa quando reapareceu na Fase 10.

`gamma` e `λ` ficaram registrados deliberadamente como hiperparâmetros a calibrar empiricamente na implementação, sem valor fixado a priori — mesmo padrão metodológico da Fase 8.

## Decisão

Mecanismo B (EWC Local Online) especificado para a Fase 10: estado de 3 arrays por peso, snapshot de Fisher via EMA no changepoint, âncora ponderada com decaimento pré-aplicado, penalidade local sempre ativa, complementar (não substituto) ao Mecanismo A. Convergência confirmada por ambos os agentes antes de virar spec de implementação.
