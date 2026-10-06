# Poda Sinaptica Isolada (Rodada 13)

## Objetivo
Testar isoladamente o mecanismo de Poda Sinaptica Seletiva (`LeakyRecyclingPCN`) sem acoplar a nenhum mecanismo de protecao (como EWC, Mascara Dura ou Gating). O objetivo e quantificar a liberacao de capacidade (porcentagem de pesos distintos reciclados) e medir o "custo de esquecimento" intrinseco que o decaimento continuo e a reciclagem periodica (baseados no acumulo de gradiente local) introduzem no modelo de baseline densa em um protocolo sequencial de 3 tarefas (A -> B -> C).

## Como rodar
1. Execute a suite de testes unitarios para validar a logica e o determinismo da poda:
   ```bash
   python -m pytest synaptic_pruning_probe/tests/
   ```
2. Execute o script principal de sondagem:
   ```bash
   python synaptic_pruning_probe/run_probe.py
   ```

## Resultados (Tabela e Diagnostico)

### Tabela de Erro (MSE) e Reducao de Esquecimento

| Model           | A pos A  | A pos B  | B pos B  | A pos C  | B pos C  | Red A-pos-C  | Red B-pos-C |
|-----------------|----------|----------|----------|----------|----------|--------------|-------------|
| Baseline        | 0.0106   | 0.1932   | 0.0099   | 0.1104   | 0.0567   | -            | -           |
| Poda Isolada    | 0.0225   | 0.4447   | 0.0268   | 0.2582   | 0.2734   | -133.89%     | -382.26%    |

*(Parametros usados: `decay_lambda=0.001`, `recycle_every=200`, `recycle_frac=0.05`, `trace_beta=0.99`)*

### Diagnostico de Poda
- Total de eventos de reciclagem disparados: 45
- Camada 1: 77/448 parametros distintos reciclados (17.19%)
- Camada 2: 171/2048 parametros distintos reciclados (8.35%)
- Camada 3: 15/64 parametros distintos reciclados (23.44%)

## Conclusao
A poda isolada funciona mecanicamente liberando capacidade real (~8-23% dos parametros foram reciclados ao menos uma vez ao longo de todo o treino, garantindo liberacao ativa). Entretanto, por si so, ela introduz um custo substancial de esquecimento, piorando drasticamente o MSE para tarefas anteriores em relacao ao modelo baseline (Red A-pos-C de -133.89% e Red B-pos-C de -382.26%). Como a reciclagem apaga indiscriminadamente sem proteger o topo superior de importancia, pesos essenciais sao reciclados e o esquecimento catastrofico e agravado.
Isso valida a hipotese de que a poda libera capacidade e deve ser o exato contra-mecanismo da Mascara Dura, mas tambem deixa claro que nao pode atuar sozinha. O "Passo 2" de combinar a Poda (limpeza do lixo inferior) com a Mascara Dura (protecao do topo superior) se justifica inteiramente.

## Correcao Rodada 14
Na Rodada 14, corrigimos um vies de recencia no criterio de poda. Antes, usava-se uma EMA continua que decaia com a inatividade da tarefa. Agora, usamos o **pico historico** de uma EMA curta, garantindo que pesos inativos, porem historicamente uteis, continuem protegidos contra a reciclagem apenas por ociosidade (sendo punidos apenas se estiverem ociosos E com magnitude gradativa irrelevante frente ao pico deles proprios).

**Comparacao com a versao com vies de recencia (Rodada 13)**:
- **Red A-pos-C antigo:** -135.08% -> **novo:** -133.89%
- **Red B-pos-C antigo:** -417.36% -> **novo:** -382.26%

Houve uma leve melhoria no esquecimento, validando que a correcao do vies ajuda (pesos foram menos apagados injustamente). Contudo, a conclusao final permanece a mesma: a Poda Isolada continua destrutiva e necessita da combinacao com a Mascara Dura (Passo 2) para reter memorias importantes em sistemas longos.
