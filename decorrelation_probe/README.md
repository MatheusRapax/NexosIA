# Teste Barato: Inibição Lateral vs L1 (Rodada 12)

## Objetivo
Medir diretamente se um termo de inibição lateral competitiva na energia do PCN decorrelaciona representações ocultas entre duas tarefas distintas (Região A e Região B) mais do que uma penalidade L1 independente e mais do que o baseline (sem modificação). O teste usou o cenário simplificado de treinar sequencialmente em A e depois em B, medindo as ativações da camada oculta 1 para um lote das regiões A e B e calculando métricas de sobreposição (Similaridade de Cosseno e Índice de Jaccard).

## Como rodar
1. Execute a suíte de testes (comprova a equivalência quando parâmetros zerados e o determinismo):
   ```
   python -m pytest decorrelation_probe/tests/
   ```
2. Execute o script principal:
   ```
   python decorrelation_probe/run_probe.py
   ```

## Resultado

| Model                | Cosine Sim   | Jaccard Index |
|----------------------|--------------|---------------|
| Baseline             | 0.6788       | 0.0323        |
| L1 (0.01)            | -0.3794      | 0.0667        |
| Lateral Inh (0.01)   | 0.9197       | 0.1429        |

*(Nota: o Jaccard Index usou um limiar correspondente ao percentil 75 da magnitude das ativações das duas tarefas)*

## Conclusão
A inibição lateral com `gamma=0.01` **não** reduziu a sobreposição em comparação ao baseline; pelo contrário, aumentou a Similaridade de Cosseno para 0.9197 e a sobreposição pelo Jaccard Index. Em contrapartida, a penalidade L1 (`lambda=0.01`) reduziu significativamente a correlação direcional (Cosine Sim negativo).
Portanto, a Hipótese E v2 (inibição lateral conforme proposta) não justifica a abertura de uma nova fase de Continual Learning neste formato, pois falhou no critério principal de decorrelacionar as representações mais do que as alternativas mais simples.

## Nota metodológica
A ativação medida no cálculo da métrica de sobreposição reflete a dinâmica de inferência PADRÃO sobre os pesos já treinados, ou seja, o termo extra (L1 ou inibição lateral) **NÃO** é reaplicado durante a medição. Assim, o resultado numérico reflete estritamente o efeito permanente do termo de penalidade sobre a REPRESENTAÇÃO APRENDIDA (pesos), não medindo uma eventual supressão ativa ou temporária do overlap operada em tempo de inferência.
