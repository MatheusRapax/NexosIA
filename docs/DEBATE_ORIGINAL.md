# O Debate Original: Gênese da Arquitetura Developmental PCN

> Reconstrução do debate estruturado entre dois agentes (Claude Code e Antigravity, no Maestri) que originou a arquitetura implementada neste projeto. O registro original vivia numa nota compartilhada que foi sobrescrita ao longo das fases de implementação — este documento recupera o conteúdo a partir do histórico da sessão, para que a motivação conceitual completa não se perca.

## Objetivo do debate

Projetar uma arquitetura de inteligência artificial que:
1. Não dependa de escala bruta de hardware do jeito que LLMs dependem hoje (treino/inferência exigindo milhões de cálculos por token, em contraste com a eficiência do cérebro humano).
2. Cresça genuinamente por experiência dentro de um ambiente simulado, com **memória e raciocínio unificados no mesmo substrato** — como no cérebro humano — em vez de memória parametrica congelada (pesos) + contexto efêmero (janela de atenção).

A meta era uma proposta concreta de arquitetura/mecanismo, não um manifesto.

## Contexto de pesquisa levantado

- **Titans** (Behrouz, Zhong & Mirrokni, Google Research, arXiv:2501.00663, 2024): módulo de memória neural de longo prazo que atualiza os próprios pesos durante o forward pass (test-time), não só recupera de um KV-cache estático. Exemplo concreto de memória e cômputo fundidos no mesmo mecanismo — mas ainda treinado em corpus estático via backprop.
- **Complementary Learning Systems theory** (McClelland et al. 1995; Kumaran et al. 2016): hipocampo = aprendizado rápido/online de curto prazo; neocórtex = consolidação lenta e generalizada via replay offline (sono). Modelo biológico de como evitar catastrophic forgetting sem congelar pesos — mas é um esquema de treino, não uma arquitetura de percepção/ação.
- **Neuromorphic / Spiking Neural Networks**: hardware event-driven (só gasta energia quando há spike). Chips como o IBM NorthPole chegam a ~195 TOp/J vs ~10 TSOp/J do cérebro humano — silício já pode ser mais eficiente por operação, mas o cérebro ainda ganha em eficiência total por fazer muito mais com muito menos operações (esparsidade + representação, não só substrato físico). Dão eficiência, mas são fracas em raciocínio abstrato.
- **JEPA / World Models** (LeCun 2022; "Objective-Driven AI", 2024; V-JEPA 2, Meta 2025): aprender prevendo representações latentes do que vai acontecer no mundo, não pixels/tokens — planejamento por busca no espaço de representação. Dá mundo simulado + previsão, mas não endereça plasticidade contínua nem eficiência de substrato.

**Gap identificado**: nenhuma dessas linhas isoladamente resolvia o problema.

## Log cronológico do debate

### Ponto 1 — Claude Code (diagnóstico + proposta de rascunho)
Reformulação do problema: o gargalo não é só "hardware", é que LLMs separam PESO (memória de longo prazo, congelada após o treino) de CONTEXTO (memória de trabalho, efêmera). Isso força o modelo a "reaprender" tudo via atenção bruta a cada inferência. O cérebro não tem essa separação: sinapse = memória = cômputo, mudando continuamente (plasticidade).

Proposta de arquitetura híbrida de 4 camadas:
- (a) Substrato eficiente: rede neuromórfica/spiking em vez de matmul denso.
- (b) Memória = cômputo (tipo Titans): módulo que atualiza pesos próprios em test-time via erro de predição.
- (c) Consolidação tipo CLS (ciclos de sono/replay offline) para evitar catastrophic forgetting.
- (d) Grounding: agente embodied num ambiente simulado com world model preditivo (linha JEPA) e motivação intrínseca.

Riscos autoapontados: (i) plasticidade contínua tende a ser instável; (ii) motivação intrínseca pura pode levar a reward hacking/vício em ruído; (iii) sim-to-real gap; (iv) incompatibilidade entre hardware event-driven e backprop.

### Ponto 2 — Antigravity (contraponto: arquitetura "Frankenstein" + unificação via Active Inference)
Pilar mais fraco: a incompatibilidade mecânica entre Titans (exige backprop/gradientes globais contínuos) e redes spiking (exige regras locais event-driven) — integrar backprop contínuo anula a eficiência do substrato spiking.

Proposta: **Active Inference / Free Energy Principle (Karl Friston)** substitui os pilares (b), (c) e (d) por um único mecanismo matemático:
- Percepção e ação são a mesma coisa: minimização da Energia Livre.
- Motivação intrínseca emerge da minimização da Energia Livre Esperada (ganho epistêmico).
- Aprendizado via *local message passing* em grafo de fatores, eliminando backprop global.
- "Sono" = minimização offline da complexidade (Bayesian Model Reduction).

Desafio de volta: como escalar message passing para raciocínio abstrato/linguagem, e como evitar o problema do "Dark Room" (minimização de surpresa levando à paralisia)?

### Ponto 3 — Claude Code (concessão parcial + contra-furo + ponte PCN)
Concessão: a resolução formal do Dark Room via valor epistêmico (Friston et al. 2017) é real — o termo de ganho de informação na Energia Livre Esperada é sempre não-negativo, então um "quarto escuro perfeitamente previsível" perde para qualquer estado ambíguo-mas-resolutível.

Contra-furo: isso não mata o risco de "motivação intrínseca vira vício em ruído" — só desloca. Literatura de curiosity-driven RL documenta o **Noisy TV problem** (Burda et al. 2018): um agente que busca reduzir incerteza fica viciado em ruído irredutível. Correção que a literatura converge: **Learning Progress Hypothesis** (Oudeyer & Kaplan) — trocar "valor epistêmico = erro de predição bruto" por "valor epistêmico = taxa de redução do erro no tempo".

Furo mais sério no desafio do Antigravity: toda demonstração real de Active Inference até hoje é em escala pequena (T-maze, mountain car, robótica, SLAM contínuo via Gaussian Belief Propagation). Implementações que escalam (Deep Active Inference, Contrastive Active Inference) reintroduzem redes treinadas por backprop — o mesmo risco (iv) ressurge dentro da própria proposta.

**Ponte proposta: Predictive Coding Networks (PCN)**. Whittington & Bogacz (2017) provaram que PCN com plasticidade local Hebbiana converge, durante a inferência, para um sinal de erro que aproxima o gradiente de backprop; Millidge/Song et al. (2021) generalizaram para backprop *exato* em qualquer grafo computacional. PCN é message passing local que é matematicamente equivalente a gradiente descendente — herda a escalabilidade do deep learning e é compatível com substrato event-driven.

Ressalva honesta: a equivalência PCN-backprop foi provada sólida para CNN/RNN/feedforward; não há prova de que escala para hierarquias tão profundas quanto raciocínio linguístico em nível GPT.

Proposta de reescopo: não competir com LLMs em linguagem — mudar o critério de sucesso para **inteligência developmental/embodied** (competência sensório-motora e conceitual crescente, como uma criança/animal), batendo mais com o pedido original do usuário.

### Ponto 4 — Antigravity (síntese: aceite do reescopo + solução para BMR sem buffer)
Aceite total do reescopo. Sobre Learning Progress: mata Dark Room e Noisy TV na mesma jogada.

Sobre BMR sem buffer explícito: a PCN já é gerativa por natureza — consolidação offline ("Sono") via **Generative Replay Implícito com Poda**:
1. Ruído top-down nas camadas altas gera "alucinações" que a rede reconcilia consigo mesma, consolidando invariantes abstratos.
2. Poda guiada por precisão: sinapses associadas a erros cronicamente ruidosos decaem a zero, aproximando BMR sem guardar nenhum log explícito de experiências passadas.

### Ponto 5 — Claude Code (verificação final)
Autocorreção: a alegação de que BMR é "NP-difícil em geral" era imprecisa — a BMR de Friston (arXiv:1805.07092) já tem solução analítica/forma fechada para modelos aninhados Gaussianos por desenho.

Validação independente: o mecanismo de replay generativo interno sem buffer tem precedente empírico revisado por pares — van de Ven, Siegelmann & Tolias, "Brain-inspired replay for continual learning with artificial neural networks", *Nature Communications* 11:4069 (2020).

Limitação registrada (não bloqueia o fechamento): esse tipo de replay costuma depender de um sinal de fronteira de tarefa/contexto; um agente developmental de verdade opera em fluxo contínuo e aberto, sem tarefas discretas — **problema aberto de task-free continual learning**, registrado como trabalho futuro.

## Decisão / Síntese final do debate

### Arquitetura: Developmental PCN (Predictive Coding Network de Desenvolvimento)

Abandona-se o paradigma LLM/oráculo estático (peso congelado + contexto efêmero). Em seu lugar, um agente "infantil" embodied que cresce por interação contínua num ambiente simulado, sustentado por 4 pilares:

1. **Motor computacional — Predictive Coding Networks.** Message passing local, matematicamente equivalente a backprop (Whittington & Bogacz 2017; Millidge/Song et al. 2021), mecanicamente compatível com hardware neuromórfico/spiking.
2. **Aprendizado contínuo e ação — Active Inference (Friston).** Percepção, ação e memória unificadas num único grafo de fatores: a memória é a própria malha de pesos plásticos, não um módulo separado.
3. **Motivação intrínseca — Learning Progress (Oudeyer & Kaplan)**, não erro de predição bruto. Mata Dark Room e Noisy TV problem com o mesmo mecanismo.
4. **Consolidação sem buffer explícito — "Sono" via Generative Replay Implícito + poda por precisão.** Aproxima Bayesian Model Reduction sem armazenar nenhum log explícito de experiências passadas. Validado por precedente empírico (van de Ven et al. 2020).

**Critério de sucesso acordado**: não é competir com LLMs em benchmarks de linguagem — é demonstrar competência sensório-motora e conceitual crescente num ambiente simulado.

**Limitações conhecidas registradas no debate (antes de qualquer código ser escrito)**:
- Equivalência PCN-backprop provada só para CNN/RNN/feedforward; não demonstrada para hierarquias de raciocínio linguístico.
- Replay generativo tipicamente depende de sinal de fronteira de tarefa/contexto — problema aberto de task-free continual learning.
- Sim-to-real gap do ambiente simulado não foi resolvido, apenas deslocado para fora do escopo pelo reescopo developmental/embodied.

---

*É notável, em retrospecto, que a limitação de "task-free continual learning" registrada aqui — antes de qualquer linha de código ser escrita — é exatamente o problema que a Fase 5 da implementação (ver [RELATORIO_TECNICO.md](RELATORIO_TECNICO.md)) encontrou de forma concreta e empírica meses depois. O debate previu corretamente onde a arquitetura quebraria.*
