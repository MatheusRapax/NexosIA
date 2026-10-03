# Debate — Rodada 2: Atacando o Bloqueio da Fase 5

> Segunda rodada de debate estruturado entre Claude Code e Antigravity, convocada especificamente para analisar o fracasso da Fase 5 (ver [RELATORIO_TECNICO.md](RELATORIO_TECNICO.md) §4.6) e propor um mecanismo substituto. Ver [ARTIGO.md](ARTIGO.md) §6-7 para os resultados experimentais (Fases 6 e 7) que validaram e testaram essa proposta.

## Objetivo da rodada

Diagnosticar com rigor por que o `SleepConsolidator` original (uma única estatística gaussiana via EMA rápido) falhou em proteger conhecimento antigo contra esquecimento catastrófico, e propor em conjunto um mecanismo substituto, com pesquisa de literatura nova.

## Síntese do diagnóstico

O problema é uma instância conhecida de **"generator forgetting"** em generative replay contínuo — o gerador dos sonhos esquece também, não só o modelo principal. Rahaf Aljundi et al. ("Task-Free Continual Learning", CVPR 2019) formaliza exatamente esse problema: como decidir quando e com que dados atualizar consolidação sem fronteira de tarefa explícita.

## Propostas consideradas e descartadas

1. **Protótipos múltiplos (CoPE)** — De Lange & Tuytelaars, "Continual Prototype Evolution" (ICCV 2021): manter K protótipos/centroides em vez de uma única gaussiana. **Furo:** resolve só "onde amostrar" o sonho, não conserta o "professor" (o modelo que rotula o sonho) se ele próprio não convergiu bem.
2. **Sono disparado por Bayesian Online Changepoint Detection** — Adams & MacKay (2007): monitorar o erro de predição para detectar mudança de regime estatisticamente. **Furo:** o agente usa Learning Progress, que busca ativamente estados de erro alto por desenho — BOCPD sobre erro absoluto confundiria exploração legítima intra-tarefa com uma mudança de regime real.

## Mecanismo escolhido: Dual-Weight PCN

Proposta do Antigravity, que resolve os dois furos acima com um único mecanismo, inspirado em Complementary Learning Systems / Brain-Inspired Replay (van de Ven et al. 2020):

1. **Fast Network**: cópia da PCN que aprende online, normalmente.
2. **Slow Network**: segunda cópia dos MESMOS pesos, nunca treinada em dados diretamente — só atualizada via Polyak averaging (EMA) dos pesos da Fast: `W_slow = β_slow·W_fast + (1-β_slow)·W_slow`. Mecanismo estabelecido (target networks em RL, Stochastic Weight Averaging), não é invenção ad-hoc.
3. **Gatilho de changepoint**: a cada passo, `delta = energia(Slow, x, y) - energia(Fast, x, y)`, suavizado por uma EMA escalar (`smoothed_delta`). Por ser **relativo** (dois modelos competindo no mesmo dado), resiste a exploração intra-tarefa — numa célula nova dentro da mesma região, ambas as redes erram de forma parecida; só quando a região muda de fato é que a Fast melhora rápido enquanto a Slow continua ancorada no passado, abrindo o delta.
4. **Geração dos sonhos**: ação sempre sorteada como one-hot válido (nunca gerada livremente — correção herdada do fracasso da Fase 4 tentativa 1, ver RELATORIO_TECNICO.md §4.3); `obs` amostrado de estatísticas gaussianas simples; a Slow Network roda só `predict()` para rotular o sonho ("professora" estável).

Zero buffer de qualquer tipo — só duas cópias de pesos (mesmo tamanho da rede original) e dois escalares. Resolve a tensão filosófica entre "estatística agregada legítima" e "buffer disfarçado": é inteiramente paramétrico.

## Limitação reconhecida por ambos os agentes antes de qualquer código

O trade-off velocidade-de-adaptação vs. retenção-de-memória não desaparece — muda de lugar (de estatística de observação para espaço de pesos). Para 2 regiões sequenciais deve bastar; numa vida com 3+ tarefas sucessivas, uma única Slow Network vai se tornar um meio-termo cada vez mais borrado entre tudo que já passou — saturação de capacidade teórica, análoga ao próprio sistema complementar biológico.

## Próximo passo acordado

Validar o detector de changepoint **isoladamente** (testes sintéticos baratos, sem RL) antes de reintegrar — mesma disciplina que funcionou bem nas Fases 3-4. Isso se tornou a Fase 6 do roadmap (ver ARTIGO.md §6).
