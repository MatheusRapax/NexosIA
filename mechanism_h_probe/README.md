# Mechanism H Probe (Confidence-Calibrated Forgetting)

Este probe avalia a hipótese inspirada no BayesPCN: modular o esquecimento de forma contínua usando a incerteza dos pesos (variância do gradiente) ao invés de usar uma poda discreta. O probe utiliza a infraestrutura modular `probe_harness`.

## Resultados

```text
Model                | A pos A  | A pos B  | B pos B  | A pos C  | B pos C  | Red A-pos-C  | Red B-pos-C 
----------------------------------------------------------------------------------------------------
Baseline             | 0.0106   | 0.1932   | 0.0099   | 0.1104   | 0.0567   |            - |            -
Poda (Rodada 14)     | 0.0225   | 0.4447   | 0.0268   | 0.2582   | 0.2734   |     -133.89% |     -382.26%
Mecanismo H          | 0.0275   | 0.5033   | 0.0276   | 0.2721   | 0.2717   |     -146.52% |     -379.12%
```

## Conclusões
O Mecanismo H provou-se **altamente destrutivo** e falhou em superar não apenas o Baseline, mas também a poda discreta original da Rodada 14 (Poda Leaky Recycling).
- **Desempenho Inferior à Poda em Lote:** O erro A-pos-C (0.2721) foi pior que o do mecanismo discreto (0.2582). Isso sugere que a erosão contínua a cada passo, mesmo atenuada pela alta confiança em pesos inativos, drena a capacidade da rede de maneira mais implacável do que reciclagens discretas e parciais.
- **Falha de Escopo Local:** Assim como na poda, a regulação unicamente temporal e local de cada sinapse através do decaimento em direção a zero cria instabilidade constante no ecossistema distribuído do PCN, destruindo as correlações aprendidas de memórias passadas.

O esquecimento calibrado, embora elegante teoricamente, demonstrou ser ineficaz no modelo PCN denso como mecanismo de mitigação do esquecimento catastrófico.
