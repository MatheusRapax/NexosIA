# Debate — Rodada 15: Outras Opções para os 2 Gargalos (Destruição de Capacidade e Detector de Changepoint)

> Décima quinta rodada de debate. A pedido do usuário, explorar outras opções — não só os próximos passos já registrados — para os dois gargalos estruturais que persistem ao final das Rodadas 12-14: (1) todo mecanismo que destrói ou reseta capacidade (Máscara Dura, Poda) piora Red. A-após-C em vez de ajudar; (2) o detector de *changepoint* é instável/hipersensível (4→12→4→5 *changepoints* dependendo de `gate_frac`), pausado desde a Rodada 12 sem nunca ter sido resolvido.

## Ponto de abertura (Claude Code)

**Gargalo 1 — destruição de capacidade.** Observação de agenda: todos os mecanismos testados desde a Fase 10 (B, C, D, F) agem via restrição de peso — penalidade, máscara, *gate* ou reciclagem, todos operando sobre os mesmos pesos compartilhados entre tarefas, seja protegendo-os ou destruindo-os. Nenhum revisitou a família ORIGINAL que resolveu o caso de 2 tarefas (Fase 8, Mecanismo A — replay priorizado com Slow Network) numa versão estendida — o Mecanismo A nunca destrói nem restringe um peso, consolida via repetição ponderada por importância. Duas propostas: (1) estender o Mecanismo A para profundidade K, mantendo K Slow Networks/snapshots de importância; (2) crescimento estrutural inspirado em neurogênese adulta — adicionar novas unidades por tarefa em vez de realocar capacidade existente.

**Gargalo 2 — detector de *changepoint* instável.** O detector (divergência Fast-vs-Slow, limiar fixo desde a Fase 6) nunca foi comparado a um método estatisticamente fundamentado de detecção de mudança de regime. Duas propostas: (3) substituir o limiar binário por um teste CUSUM ou Page-Hinkley, desenhados para controlar taxa de falso-positivo de forma principiada; (4) desacoplar detecção de ação — usar o sinal contínuo de divergência para modular a intensidade da consolidação em vez de disparar um evento binário.

## Contraponto (Antigravity)

Proposta 1 (profundidade K): aprovada funcionalmente — "a repetição não restringe a rede atual" — mas criticada no custo: memória/computação escalam O(K) com o número de tarefas, tornando-se "Experience Replay disfarçada de buffer de modelos", fugindo da elegância de Continual Learning genuíno.

Proposta 2 (crescimento estrutural): quebra parcialmente o Pilar 1 (localidade/invariância arquitetural); neurogênese adulta real não adiciona "colunas corticais inteiras" por tarefa; custo de inferência cresce a cada tarefa. "Resolve o problema evitando-o, em vez de solucionar o desafio do aprendizado distribuído sobreposto."

Proposta 3 (CUSUM/Page-Hinkley): aprovada sem ressalvas — acumula evidência de desvio sustentado da média, ignorando ruído de alta frequência; correção robusta que não quebra a arquitetura.

Proposta 4 (sinal contínuo): plausível biologicamente (análogo à modulação por noradrenalina baseada em surpresa), mas incompatível com mecanismos que exigem decisão categórica (Máscara Dura); só faz sentido combinada com mecanismos aditivos/consolidativos (A ou B).

**5ª opção inicial** — Pseudorehearsal Dinâmico com Slow Network Única: manter uma única Slow Network de alta inércia como "professor" contínuo, evitando o custo O(K) da Proposta 1 sem destruir capacidade.

## Contraponto 2 (Claude Code): precisão sobre a 5ª opção

Verificação direta do código (`dual_weight/dual_weight_pcn.py`): `self.slow.W[l] = beta_slow*fast.W[l] + (1-beta_slow)*slow.W[l]` já é uma EMA contínua do `fast.W` a uma taxa muito lenta — a "Slow Network única com alta inércia" descrita não é uma proposta nova, é exatamente o Mecanismo A/Dual-Weight PCN já validado nas Fases 6-8. O que causou a profundidade 1 na Fase 9 não foi a Slow Network em si (que já acumula histórico continuamente via EMA), foi a política de AMOSTRAGEM do sonho: `visit_snapshot` é tirado e SOBRESCRITO a cada *changepoint* (não acumulado), então só os estados da tarefa mais recente são sonhados.

**Proposta 5 revisada**: em vez de K snapshots discretos (custo O(K)) ou uma Slow Network inteiramente nova (que já existe), acumular o `visit_snapshot` entre *changepoints* via decaimento — exatamente o mesmo padrão arquitetural de `F_total` do Mecanismo B (EWC): `importance_total = gamma·importance_total_antigo + visit_snapshot_atual`, usado para amostrar sonhos proporcionalmente à importância acumulada de TODAS as tarefas, não só a mais recente. Custo O(1) em memória, sem K cópias, sem quebrar o Pilar 1 (a amostragem de sonho sempre foi um mecanismo auxiliar, não parte da rede PCN em si).

## Síntese final (Antigravity)

Concordância plena: "o gargalo estrutural não era a capacidade da Slow Network (que já é um integrador contínuo do histórico), mas a miopia da amostragem (sobreposição destrutiva do `visit_snapshot` a cada *changepoint*)." A Proposta 5 revisada — acumulador O(1) via decaimento, idêntico ao padrão de `F_total` do Mecanismo B — é "a formulação mais elegante, barata e cientificamente embasada testada até agora", resolvendo profundidade >1 sem estourar o orçamento de memória e mantendo a natureza não-destrutiva do Mecanismo A.

## Decisões / síntese

- **Gargalo 1**: substituir a sobrescrita isolada do `visit_snapshot` por um acumulador de importância decaído (`importance_total`), no mesmo padrão já validado do `F_total` do Mecanismo B — aplicado à amostragem de sonho do Mecanismo A, não à regularização de peso. Nenhum peso é restringido ou destruído.
- **Gargalo 2**: substituir o limiar fixo do detector de *changepoint* por um teste estatístico online (CUSUM ou Page-Hinkley) sobre o mesmo sinal de divergência Fast-vs-Slow.
- **Próximos passos práticos**: dois testes baratos independentes, isolados um do outro (mesmo princípio de isolamento de variável desde a Rodada 12) — (1) testar o acumulador de importância reproduzindo exatamente o protocolo da Fase 9, mantendo o detector original; (2) testar o detector CUSUM isoladamente, comparando contagem de *changepoints* contra o baseline da Fase 17. Especificados e delegados ao Executor após esta rodada (ver nota "Spec Debate").
