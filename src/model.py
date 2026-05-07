class SubNetwork(nn.Module):
    def __init__(self, dim):
        super(SubNetwork, self).__init__()
        hidden_dim = dim + 10
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.1),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.1),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
        
    def forward(self, x):
        return self.net(x)

class DeepBSDE(nn.Module):
    def __init__(self, config):
        super(DeepBSDE, self).__init__()
        self.config = config
        
        self.Y_0 = nn.Parameter(torch.rand(1) * 10) 
        self.Z_0 = nn.Parameter(torch.zeros(1, self.config.d))
        
        self.sub_networks = nn.ModuleList([SubNetwork(self.config.d) 
                                           for _ in range(self.config.N - 1)])
        
    def forward(self, X, dW):
        batch_size = X.shape[0]
        
        Y = self.Y_0.expand(batch_size, 1)
        Z = self.Z_0.expand(batch_size, self.config.d)
        
        for t in range(self.config.N):
            Y = Y - self.config.driver(t, X[:, t, :], Y, Z) * self.config.dt + \
                torch.sum(Z * dW[:, t, :], dim=1, keepdim=True)
            
            if t < self.config.N - 1:
                Z = self.sub_networks[t](X[:, t+1, :])
                
        return Y
