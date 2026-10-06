from decorrelation_probe.probe_pcn import LateralInhibitionProbePCN

class LateralInhibitionWindowedPCN(LateralInhibitionProbePCN):
    def __init__(self, layer_sizes, inference_lr=0.1, inference_steps=20, weight_lr=0.01, seed=0, gamma=0.01, window_steps=200):
        super().__init__(layer_sizes, inference_lr, inference_steps, weight_lr, seed, gamma)
        self.window_steps = window_steps
        self.steps_since_boundary = None  # None = nunca houve fronteira ainda, inibicao OFF

    def on_changepoint(self):
        self.steps_since_boundary = 0

    def train_step(self, x_input, y_onehot):
        active = self.steps_since_boundary is not None and self.steps_since_boundary < self.window_steps
        if self.steps_since_boundary is not None:
            self.steps_since_boundary += 1
        if active:
            return super().train_step(x_input, y_onehot)
        else:
            saved_gamma = self.gamma
            self.gamma = 0.0
            energy = super().train_step(x_input, y_onehot)
            self.gamma = saved_gamma
            return energy
