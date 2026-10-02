# Fase 5: Agente Unificado e Sono Periódico (Bloqueado - 4ª Tentativa)

Esta fase teve como objetivo integrar o agente da Fase 3 (navegação e learning progress) com o consolidador da Fase 4 (sono generativo). A proposta principal era disparar ciclos de sono *periodicamente* baseado em contagem de episódios, avançando rumo a um agente de aprendizagem contínua livre de tarefas (*task-free continual learning*).

## Diagnóstico Técnico do Bloqueio Definitivo

Apesar de múltiplas correções geométricas e calibração de hiperparâmetros (como restringir o treinamento e os objetivos para blocos densos 2x2 com `epistemic_weight=1.0`), o experimento consolidou um bloqueio fundacional intransponível sob as atuais restrições arquiteturais. 

A falha engloba duas dimensões:

1. **Ruído Absoluto x Resolução do Grid:** O bloco 2x2 conseguiu melhorar o Erro Quadrático Médio (MSE) de ~0.89 para ~0.22 durante a Fase A (treino focado). No entanto, um MSE de 0.22 gera um desvio padrão de erro (~0.47) que é superior à distância real entre as células do grid (0.25). Como a distância de predição afoga no próprio ruído, o modelo perde a bússola topológica local e o agente erra o alvo `(0,0)`, estagnando sua taxa de sucesso em ~38%.
2. **Colapso Task-Free do SleepConsolidator:** Conforme provado pelas métricas, o sono generativo fracassou categoricamente em proteger o aprendizado prévio. Quando o agente entra na Fase B, a sua tracking veloz (EMA com `alpha=0.02`) transaciona inteiramente a memória estatística da rede para a Região B. Quando ocorre o "sono" na Fase B, o modelo "sonha" com a Região B e treina usando sua própria degeneração absoluta. O resultado final é que o MSE na Região A vai para `0.88`, idêntico à condição sem sono.

Em resumo, a integração unificada online de um Predictive Coding Network com Generative Replay puro esbarra em limites de convergência rápida e perda contínua de distribuição. Como foi acordado com a Arquitetura que esta era a última tentativa, o projeto desta fase é dado como paralisado e este relatório assume-se como o achado empírico final.
