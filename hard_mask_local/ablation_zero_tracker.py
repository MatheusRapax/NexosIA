from hard_mask_local.hard_mask_tracker import HardMaskLocalTracker

class HardMaskZeroTracker(HardMaskLocalTracker):
    """Variante de diagnostico (Rodada 8 / Fase 14): zera pesos congelados no forward
    em vez de ancorar em w_anchor. Usada apenas para medir interferencia residual de
    forward vs deficit de capacidade. NAO USAR EM PRODUCAO."""
    def apply_penalty(self, W_current):
        offset = 1 if len(W_current) == self.L + 1 else 0
        for l in range(1, self.L + 1):
            W = W_current[l - 1 + offset]
            W[self.frozen_mask[l]] = 0.0
