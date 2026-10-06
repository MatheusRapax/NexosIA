# Mechanism G Probe (Context-Gated Lateral Inhibition)

Este probe utiliza um `probe_harness` genérico para testar abordagens de inibição lateral guiadas por contexto, comparando o `Baseline` (sem gate), `Gating-only` (Mecanismo D), `G-lite` (inibição temporal por janela) e `G-v2` (inibição espacial via máscara do `GatingLocalTracker`).

## Resultados

```text
Model                | A pos A  | A pos B  | B pos B  | A pos C  | B pos C  | Red A-pos-C  | Red B-pos-C 
----------------------------------------------------------------------------------------------------
Baseline             | 0.0106   | 0.1932   | 0.0099   | 0.1104   | 0.0567   |            - |            -
Gating-only          | 0.0106   | 0.3035   | 0.0150   | 0.2101   | 0.2452   |      -90.37% |     -332.40%
G-lite               | 0.0106   | 0.1967   | 0.0101   | 0.1127   | 0.0637   |       -2.12% |      -12.31%
G-v2                 | 0.0104   | 0.3556   | 0.0170   | 0.2204   | 0.2402   |      -99.71% |     -323.61%
```

## Conclusões
Nenhuma das abordagens de inibição (G-lite e G-v2) superou o Baseline ou o Mecanismo D (`Gating-only`).
1. **G-lite (inibição por janela)** teve desempenho estatisticamente equivalente (levemente pior) ao Baseline. Ele falha em proteger a memória, evidenciando que a inibição unicamente temporal não introduz separação espacial nas representações.
2. **G-v2 (inibição mascarada por gate)** conseguiu ser mais catastrófico que o Baseline (assim como o `Gating-only`), confirmando que apenas o uso do tracker/gate sem estabilizadores adicionais é altamente instável para aprendizado contínuo nesse cenário restrito (conforme observado no Mecanismo D original, que exigiu o combo B+C+D para funcionar). A adição da inibição espacial (`G-v2`) em cima do gating piorou o erro `A pos C` marginalmente (-99.71% de redução em comparação com -90.37% do Gating-only).

Assim, a inibição lateral por si só, mesmo quando restrita ao contexto, se mostra ineficaz como mecanismo autônomo.
