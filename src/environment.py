class HighDimBlackScholes:
    def __init__(self, dim=100, total_time=1.0, num_time_intervals=20):
        self.d = dim 
        self.T = total_time
        self.N = num_time_intervals
        self.dt = self.T / self.N
        
        self.mu = 0.05  
        self.sigma = 0.2  
        
        self.r_l = 0.04  
        self.r_b = 0.06  
        
        self.X_0 = torch.ones(self.d).to(device) * 100.0
        self.strike = 100.0
        
    def generate_paths(self, batch_size):
        dW = torch.randn(batch_size, self.N, self.d).to(device) * np.sqrt(self.dt)
        X = torch.zeros(batch_size, self.N + 1, self.d).to(device)
        X[:, 0, :] = self.X_0
        
        for t in range(self.N):
            X[:, t+1, :] = X[:, t, :] + self.mu * X[:, t, :] * self.dt + \
                           self.sigma * X[:, t, :] * dW[:, t, :]
        return X, dW
        
    def driver(self, t, X, Y, Z):
        hedge_value = torch.sum(Z / self.sigma, dim=1, keepdim=True)
        borrowed_amount = Y - hedge_value
        
        rate = torch.where(borrowed_amount > 0, self.r_b, self.r_l)
        return -rate * Y
        
    def terminal_condition(self, X_T):
        geometric_avg = torch.exp(torch.mean(torch.log(X_T), dim=1))
        return torch.relu(geometric_avg - self.strike)

    def analytical_pricing(self):
        sigma_hat = self.sigma / np.sqrt(self.d)
        q = 0.5 * self.sigma**2 * (1 - 1/self.d)
        
        S0 = self.X_0[0].item() 
        
        d1 = (np.log(S0 / self.strike) + (self.r_l - q + 0.5 * sigma_hat**2) * self.T) / (sigma_hat * np.sqrt(self.T))
        d2 = d1 - sigma_hat * np.sqrt(self.T)
        
        exact_price = np.exp(-q * self.T) * S0 * norm.cdf(d1) - \
                      np.exp(-self.r_l * self.T) * self.strike * norm.cdf(d2)
        return exact_price
