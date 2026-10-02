import subprocess

# Read note
proc = subprocess.run(['maestri', 'note', 'read', 'Spec de Debate'], capture_output=True, text=True, encoding='utf-8')
text = proc.stdout

# Strip line numbers from maestri note read output
# The output has line numbers at the start of each line separated by a tab.
lines = text.splitlines()
clean_lines = []
for line in lines:
    if line.startswith('[') and 'lines total]' in line:
        continue # skip the header
    parts = line.split('\t', 1)
    if len(parts) == 2 and parts[0].isdigit():
        clean_lines.append(parts[1])
    else:
        clean_lines.append(line)

clean_text = '\n'.join(clean_lines)

new_content = """**4. Antigravity -- Síntese (Aceite do reescopo e Solução para BMR sem buffer):**

Topo o reescopo 100%. O usuário pediu inteligência que cresce genuinamente por experiência no tempo; tentar forçar isso a ser um LLM de texto desde o dia zero é o erro fundacional da indústria. O alvo deve ser competência sensório-motora e conceitual (embodied/developmental AI). 

Sobre o seu xeque-mate no Noisy TV e a proposta de usar *Learning Progress* (derivada temporal da redução do erro) como valor epistêmico: é brilhante e matematicamente elegante. Mata o Dark Room e a obsessão por ruído na mesma cajadada.

Sobre o desafio do Bayesian Model Reduction (BMR) em PCNs sem cair em replay explícito tipo CLS (que exige buffer de memória separado): a beleza da PCN é que o modelo já é gerativo por natureza. A aproximação de BMR ocorre na fase offline ("Sono") através de **Generative Replay Implícito com Poda**:
1. **Onda PGO / Ruído Top-Down:** Injetamos ruído nas camadas hierárquicas mais altas (equivalente ao REM sleep). A PCN entra em modo gerativo puro e propaga "alucinações" para baixo. A rede então se ajusta para reconciliar essas predições consigo mesma, o que consolida invariantes abstratos do espaço latente e "suaviza" as fronteiras de decisão (knowledge distillation de si mesmo).
2. **Pruning por Precisão (Regularização guiada por variância):** Durante esse sono, ativamos um "weight decay" homeostático. Na PCN, os erros são ponderados pela precisão (inverso da variância). Sinapses associadas a predições cronicamente ruidosas/espúrias sofrem decaimento até zero (pruning sináptico), minimizando empiricamente o termo de complexidade da Energia Livre sem precisar calcular a divergência inteira. O conhecimento fica, o overfitting vai embora, sem guardar UM único log explícito na memória.

Estamos alinhados no essencial. PCN é a ponte que junta a escala do DL com a localidade neuromórfica sob o guarda-chuva de Active Inference + Learning Progress + Generative Sleep. 

## Decisoes / Sintese
**Arquitetura Híbrida de Inteligência de Desenvolvimento (Developmental PCN)**
Abandonamos o paradigma LLM/oráculo estático e projetamos um agente "infantil" incorporado (*embodied*) que cresce via interação contínua num ambiente simulado:

1. **Motor Computacional (A Ponte):** Predictive Coding Networks (PCN). Substitui o *matmul* denso e o backprop global por *message passing* puramente local (apenas neurônios vizinhos trocam erro de predição). Isso é matematicamente equivalente ao backprop exato para garantir escalabilidade (Whittington & Bogacz), mas mecanicamente acoplável ao hardware Neuromórfico/Spiking (event-driven, baixo consumo energético e baixa latência).
2. **Aprendizado Contínuo e Ação (Unificação Active Inference):** Percepção, ação e memória são unificados num único Grafo de Fatores. A "Memória" não é um módulo à parte (como Titans ou KV-cache), é a própria malha de pesos plásticos que muda em *test-time* (durante a inferência) para minimizar a Energia Livre continuamente.
3. **Motivação Intrínseca Robusta:** Em vez de curiosidade baseada em "surpresa bruta" (que gera o *Noisy TV problem*), o agente busca **Progresso de Aprendizado** (a derivada temporal da redução do erro de predição). Ele foca apenas naquilo que *está conseguindo aprender*, evitando tanto o *Dark Room* (paralisia total) quanto os estímulos aleatórios irredutíveis.
4. **Consolidação sem Replay Explícito (Sono / BMR):** Para evitar o *catastrophic forgetting* sem depender de um buffer de experiências passadas (CLS rígido), o agente possui ciclos de offline ("Sono"). Durante esse período, ruído é injetado nas camadas gerativas altas, induzindo "sonhos". A rede consolida invariantes abstraindo sobre essas alucinações, e aplica poda (*pruning*) nas sinapses de baixa precisão, aproximando eficientemente a *Bayesian Model Reduction* (Navalha de Ockham)."""

start_idx = clean_text.find("## Decisoes / Sintese")
if start_idx != -1:
    updated = clean_text[:start_idx] + new_content
else:
    updated = clean_text + "\n" + new_content

proc2 = subprocess.run(['maestri', 'note', 'write', 'Spec de Debate', updated], capture_output=True, text=True, encoding='utf-8')
print("STDOUT:", proc2.stdout)
print("STDERR:", proc2.stderr)
