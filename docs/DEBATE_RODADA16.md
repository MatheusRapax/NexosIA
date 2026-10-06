# Debate — Rodada 16: Harness de Teste Rápido + Ideias Concretas da Literatura PCN

> Décima sexta rodada de debate. A pedido do usuário, após uma pesquisa de literatura (não mais inspiração neurocientífica genérica, mas papers que já resolveram parcialmente o MESMO problema — esquecimento catastrófico — na MESMA família de arquitetura, PCN com regras locais): construir um harness reutilizável de teste barato, para testar dezenas de hipóteses rapidamente antes de acoplar qualquer uma na arquitetura principal, e usar esse harness para testar duas ideias concretas: competição lateral gateada por contexto (Ororbia & Mali, *Lifelong Neural Predictive Coding*, NeurIPS 2022) e esquecimento calibrado por incerteza (BayesPCN, NeurIPS 2022).

## Ponto de abertura (Claude Code)

Proposta de harness `probe_harness/` com três peças reutilizáveis: `transitions.py` (gera as transições padrão A/B/C, reusando `integration_v8` via import), `train_and_measure.py` (treino supervisionado direto A→B→C, sem *loop* de RL — o mesmo protocolo mais barato já usado nas Rodadas 12-14 — retornando os 5 MSEs padrão) e `report.py` (tabela comparativa padrão com Red.A-após-C/Red.B-após-C). Qualquer hipótese nova passaria a precisar só de uma subclasse de PCN (o único código realmente novo) mais três linhas de orquestração, em vez de um módulo inteiro por rodada.

Primeira hipótese proposta para o harness — Mecanismo G (competição lateral gateada): em vez do termo incondicional e sempre-ativo da Hipótese E v2 (Rodada 12, refutada — aumentou a correlação em vez de reduzir), gatear a competição por uma janela temporal pós-*changepoint* (análoga ao padrão "protecting" já usado em outros mecanismos do projeto).

## Contraponto (Antigravity)

Concordância plena com o design do harness — elimina o *boilerplate* que vinha inflando as Rodadas 12-14.

Mas crítica de fundo à formulação inicial do Mecanismo G: gatear a inibição por uma janela temporal pós-*changepoint* esmaga retrospectivamente representações já densas, não as decorrelaciona durante o aprendizado — diferente do mecanismo real de Ororbia & Mali, onde a competição lateral associada ao contexto constrói representações esparsas e ortogonais *enquanto* a tarefa é aprendida, roteando o aprendizado para sub-redes distintas. Para ser fiel à ideia de "roteamento por contexto decide quem compete", o sinal deveria modular QUAIS unidades competem durante o aprendizado contínuo, não atuar como um interruptor temporal de consolidação. Ainda assim, validou o valor de testar a versão temporal isoladamente, como controle: confirma ou refuta se "estar sempre-ativo" sozinho já explicava o fracasso da Rodada 12. Sobre localidade (Pilar 1): o termo de soma instantânea entre unidades da mesma camada é aceitável sob a mesma premissa já debatida na Rodada 12 (um "pool de interneurônios inibitórios globais", broadcast neuromodulatório local à camada).

## Síntese (Claude Code): Mecanismo G v2, reusando infraestrutura já validada

Em vez de inventar um sinal de contexto novo, reusar o `GatingLocalTracker` (Mecanismo D, já validado desde a Fase 14) como fonte do sinal: ele já mantém, por camada, uma máscara binária `gate[l]` indicando quais unidades estão ativas para a tarefa atual. A competição lateral passa a atuar SÓ dentro do subconjunto marcado como ativo pelo gate, a cada passo de aprendizado (não só pós-*changepoint*): `dx[l] -= gamma * gate[l] * (soma(x[l]*gate[l]) - x[l]*gate[l])`. Unidades fora do gate atual não sofrem pressão competitiva — ficam protegidas por não estarem em uso para essa tarefa. Isso é uma restrição ESPACIAL (quais unidades competem), não temporal, decidida pelo próprio mecanismo de roteamento por contexto que o projeto já tem e já validou isoladamente.

## Síntese final (Antigravity)

Concordância plena: a formulação v2 captura a essência do "roteamento por contexto" sem violar o Pilar 1 e sem introduzir hiperparâmetros exógenos injustificados — a inibição torna-se dependente de estado de forma espacial, e unidades não alocadas para a tarefa atual ficam blindadas tanto contra atualização de peso (Mecanismo D original) quanto contra a supressão dinâmica de ativação (Mecanismo G v2). Concordância em testar as duas variantes no harness: G-lite (janela temporal, controle de ablação — isola se "sempre-ativo" bastava para explicar a Rodada 12) e G-v2 (subconjunto espacial via *gate*, a hipótese real de decorrelação guiada por contexto).

## Decisões / síntese

- **Harness `probe_harness/`**: três módulos reutilizáveis (transições padrão, treino-e-medição supervisionado direto, tabela comparativa) para reduzir o custo de especificar/verificar cada nova hipótese a uma subclasse de PCN + orquestração mínima.
- **Mecanismo G-lite**: inibição lateral incondicional (mesmo termo da Hipótese E v2), mas gateada por uma janela temporal pós-*changepoint* — controle de ablação, não a hipótese principal.
- **Mecanismo G-v2**: inibição lateral restrita ao subconjunto de unidades marcadas ativas pelo `gate[l]` do `GatingLocalTracker`, atuando a cada passo durante o aprendizado — a hipótese fiel ao "roteamento por contexto" de Ororbia & Mali, reusando infraestrutura já validada (Mecanismo D) em vez de um sinal de contexto novo.
- **Próximo passo prático**: especificar e delegar o harness junto com as duas variantes de teste, testadas em paralelo no mesmo protocolo A→B→C barato. Esquecimento calibrado por incerteza (BayesPCN) fica registrado como próxima hipótese na fila, após este teste.
