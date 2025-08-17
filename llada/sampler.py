import torch


class Sampler:
    def __init__(self, length=256, window=None,**kargs):
        self.length = length
        self.window = window
        self.kargs = kargs    
        self._pdf = None

        assert self.window <= self.length
        
    def pdf():
        pass
       
    def sample(self, src: torch.Tensor):
        '''
        rejection sampling
        '''

        uniform = torch.rand(src.shape[0], device=src.device)  # Generate uniform random numbers on the same device as src
        # print("look: ", len(self.pdf()[:src.shape[0]]), self.pdf()[:src.shape[0]])
        return src[uniform < self.pdf()[:src.shape[0]]] 


class GaussianSampler(Sampler):
    def __init__(self, length=256, window=None,sigma=1.0, scale=1.0):
        super().__init__(length,window)
        self.sigma = sigma
        self.scale = scale

    
    def pdf(self):
        '''
        Generate Gaussian PDF values.
        '''
        if self._pdf is not None:
            return self._pdf
        mean = 0.0  
        std_dev = 1

        x = torch.linspace(mean, mean + self.sigma * std_dev, self.window, device='cuda')

        self._pdf = self.scale * torch.exp(-0.5 * ((x - mean) / std_dev) ** 2) / (std_dev * torch.sqrt(torch.tensor(2 * torch.pi)))
        extended_pdf = torch.zeros(self.length-self.window, device='cuda')
        self._pdf = torch.cat([self._pdf, extended_pdf])

        return self._pdf
    
class UniformSampler(Sampler):
    def __init__(self, length=256, window=None, number=0):
        super().__init__(length,window)
        self.number = number
        assert self.number <= self.length
    

    def sample(self, src: torch.Tensor):
        if self.number >= src.shape[0]:
            return src
        
        indices = torch.sort(torch.randperm(min(src.shape[0],self.window))[:self.number]).values
       
        return src[indices]
