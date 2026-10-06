# Debate — Rodada 18: Mecanismo I — Alocação Fixa e Disjunta de Capacidade por Tarefa

> Décima oitava rodada de debate. Último item da fila das Rodadas 12-17: testar capacidade por ALOCAÇÃO, nunca destrutiva — em vez de qualquer forma de restrição/destruição/decaimento de peso (seis famílias já refutadas), reservar sub-redes/sub-espaços fixos por tarefa desde o início, nunca realocados nem decaídos.

## Ponto de abertura (Claude Code)

Análise do código do Mecanismo D (`GatedPredictiveCodingNetwork`): o *masking* (`e[l] *= gate[l]`, `x[l] *= gate[l]`) já garante, estruturalmente, que enquanto o *gate* de uma unidade permanecer 0, nem os pesos de entrada nem os de saída daquela unidade recebem qualquer atualização — já é, por construção, uma proteção perfeita de peso ENQUANTO o *gate* for fixo. A fragilidade de todas as fases anteriores nunca foi o mecanismo de mascaramento (que protege bem), foi a REALOCAÇÃO do *gate* ao longo do tempo (dinâmica, sujeita a esgotamento de capacidade e colisão entre tarefas, Fases 14-17).

Proposta — Mecanismo I: reusar `GatedPredictiveCodingNetwork` sem mudar seu masking, só trocar como o *gate* é decidido — particionar cada camada oculta em 3 blocos disjuntos de tamanho igual, um por tarefa, fixados uma vez (sem sorteio, sem histórico, sem fallback de esgotamento, já que os 3 blocos juntos cobrem toda a capacidade sem *overlap*). Protocolo de 3 cenários com a MESMA fração de capacidade (1/3) para isolar a variável real (fixo vs. dinâmico, não fração de capacidade): Baseline, Gating-dinâmico (`gate_frac=1/3`), Mecanismo I (alocação fixa).

## Contraponto (Antigravity)

Concordância com o desenho de isolamento de variável — se o *gating* dinâmico com `gate_frac=1/3` falhar e o Mecanismo I funcionar, prova-se que o problema nunca foi escassez de capacidade, mas a sobreposição estatística inevitável da alocação dinâmica. Sobre risco de capacidade insuficiente (1/3 de 64 e 32 unidades = camadas de 21 e 10): o termômetro empírico é o próprio "A pós A" — se comparável ao baseline, a capacidade basta; senão, escalar a rede é o próximo passo, não testar múltiplas frações agora. Sem violação de localidade.

Identificou um detalhe prático crítico: avaliar a Tarefa A com o *gate* de A restaurado, mesmo após treinar B/C, exige um "oráculo de tarefa" no momento da inferência — o `train_and_measure` genérico do `probe_harness` assume um cenário "cego" (usa sempre o último *gate* ativo), então avaliar A sob o *gate* de B daria erro quase total. Propôs inicialmente um laço de treino/medição bespoke, fora do harness.

## Síntese (Claude Code): estender o harness em vez de contornar

Em vez de um laço bespoke (que reduziria o valor do harness para hipóteses futuras com a mesma necessidade), estender `train_and_measure` com um parâmetro opcional novo e retrocompatível, `task_gate_fn` (`None` por padrão, preservando o comportamento de `mechanism_g_probe`/`mechanism_h_probe` inalterado): quando fornecido, troca o *gate* explicitamente antes de cada avaliação (`evaluate(task_name, transitions)`) e antes de cada retomada de treino após uma fronteira (`signal_boundary(next_task)`), sem fuga entre avaliação com oráculo e o *gate* de treino correto.

## Síntese final (Antigravity)

Concordância plena — a extensão mantém a generalidade do harness (suporta tanto Task-IL/oráculo quanto Domain-IL "cego", com retrocompatibilidade total), garantindo que a rede esteja sempre no contexto correto tanto para treino quanto para avaliação multitarefa.

## Decisões / síntese

- **Mecanismo I**: partição fixa e disjunta de cada camada oculta em 3 blocos (um por tarefa), decidida uma vez, nunca realocada; `set_task_gate(task_id)` apenas troca `self.gate[l]` para o bloco daquela tarefa, reusando o masking já validado do Mecanismo D sem modificá-lo.
- **Extensão do `probe_harness`**: `train_and_measure` ganha o parâmetro opcional `task_gate_fn`, retrocompatível, permitindo avaliação com oráculo de tarefa para qualquer mecanismo futuro que precise — não só para o Mecanismo I.
- **Protocolo de teste**: 3 cenários com a mesma fração de capacidade (1/3) — Baseline, Gating-dinâmico, Mecanismo I — isolando fixo-vs-dinâmico como a única variável.
- **Próximo passo prático**: especificar e delegar a extensão do harness junto com o Mecanismo I.
