class PageHinkleyDetector:
    def __init__(self, delta_tol: float = 0.005, lambda_threshold: float = 0.05):
        self.delta_tol = delta_tol
        self.lambda_threshold = lambda_threshold
        self.mean_estimate = 0.0
        self.n = 0
        self.m_t = 0.0
        self.M_t = 0.0

    def update(self, x: float) -> bool:
        self.n += 1
        self.mean_estimate += (x - self.mean_estimate) / self.n
        self.m_t += (x - self.mean_estimate - self.delta_tol)
        self.M_t = min(self.M_t, self.m_t)
        ph_stat = self.m_t - self.M_t
        if ph_stat > self.lambda_threshold:
            self.mean_estimate = 0.0
            self.n = 0
            self.m_t = 0.0
            self.M_t = 0.0
            return True
        return False
