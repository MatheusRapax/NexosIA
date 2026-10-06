# Debate — Rodada 17: Mecanismo H — Esquecimento Calibrado por Confiança (inspirado em BayesPCN)

> Décima sétima rodada de debate. Próxima hipótese da fila registrada na Rodada 16: testar, usando o `probe_harness` já construído, a ideia central do BayesPCN (NeurIPS 2022) — esquecimento como um decaimento tipo difusão guiado por incerteza calibrada, não um limiar cru de magnitude/gradiente acumulado como a poda das Rodadas 13-14.

## Ponto de abertura (Claude Code)

Proposta — Mecanismo H: reusar a infraestrutura já validada (EMA de curto prazo + `historical_peak`, Rodada 14) e acrescentar um segundo acumulador, a variância do gradiente local. Formulação inicial: `grad_mean`/`grad_var` calculados sobre `abs(last_local_grad)`; `confidence = historical_peak / (sqrt(grad_var) + epsilon)`; `decay_rate = decay_lambda / (1 + confidence)`; `W[l] *= (1 - decay_rate)` todo passo, contínuo, sem evento discreto de reciclagem nem reinicialização Xavier em lote — diferente da poda das Rodadas 13-14 em quatro pontos: decaimento contínuo (não em lote), taxa modulada por peso individual (não corte binário de percentil), sempre em direção a zero (nunca substituição abrupta de valor), e "confiança" usa variância (não só magnitude).

## Contraponto (Antigravity): defeito no uso de magnitude para a variância

Concordância com a fidelidade geral ao BayesPCN — o decaimento modulado por precisão (inverso da variância) imita o comportamento de um prior Bayesiano, de forma local e O(1) em memória por sinapse.

Mas identificou um defeito lógico crítico: calcular média/variância sobre `abs(last_local_grad)` em vez do gradiente bruto (com sinal) falha exatamente no caso que deveria capturar. Um peso recebendo gradiente oscilante entre +1 e -1 a cada passo (incerteza máxima, o pior caso) teria `abs(gradiente)` constante em 1 — a média convergiria para 1 e a variância para 0, dando confiança INFINITA ao peso mais ruidoso da rede. A correção: calcular `grad_mean`/`grad_var` sobre o gradiente bruto, sem `abs()` — um gradiente oscilante gera média próxima de zero (cancelamento) e variância alta (desvios reais de ±1), produzindo confiança baixa, como deveria ser. `historical_peak` passa a usar `abs(grad_mean)` (do valor médio, não da magnitude instantânea).

Verificação de casos-limite (ambos corretos com a correção): peso nunca usado — `historical_peak=0`, `confidence=0`, `decay_rate` no máximo (`decay_lambda`), decai à taxa máxima, como esperado. Peso historicamente importante mas agora inativo — gradiente atual consistentemente perto de zero, variância recente cai, `confidence` explode (`historical_peak` fixo dividido por variância pequena), `decay_rate` cai a zero — protegido, sem herdar o viés de recência da Rodada 13 (`historical_peak` nunca decai por inatividade, mesma garantia já validada).

Confirmação de localidade (Pilar 1): todas as variáveis novas têm o formato exato de `W[l]`, todas as operações são elemento-a-elemento — mecanismo 100% distribuído, sem nenhum sinal global.

## Síntese (Claude Code)

Aceitação integral da correção — usar o gradiente bruto evita exatamente o defeito introduzido na proposta original (confiança falsa para gradiente oscilante).

## Decisões / síntese

- **Mecanismo H**: `grad_mean`/`grad_var` calculados sobre o gradiente local bruto (com sinal), não sua magnitude; `historical_peak = max(historical_peak, abs(grad_mean))`; `confidence = historical_peak / (sqrt(grad_var) + epsilon)`; `decay_rate = decay_lambda / (1 + confidence)`; decaimento contínuo `W[l] *= (1 - decay_rate)` todo passo — sem evento discreto, sem reset em lote, sem reinicialização Xavier abrupta.
- **Diferença de família em relação à poda (Rodadas 13-14)**: decaimento suave e por peso individual, guiado por confiança calibrada (magnitude histórica relativa à variância recente), nunca um corte de percentil nem uma reinicialização em lote — testa diretamente se a causa da falha da poda era o CRITÉRIO (resolvido nas Rodadas 13-14) ou a NATUREZA ABRUPTA/EM LOTE do mecanismo de reciclagem em si.
- **Próximo passo prático**: especificar e delegar o Mecanismo H usando o `probe_harness` (Rodada 16), comparando contra o baseline denso e a poda já validada (Rodadas 13-14) no mesmo protocolo A→B→C.
