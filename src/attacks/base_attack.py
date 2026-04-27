class BaseAttack:
    def __init__(self, model, data, device):
        self.model = model
        self.data = data
        self.device = device
        self.model.eval()

    def attack(self, *args, **kwargs):
        raise NotImplementedError
