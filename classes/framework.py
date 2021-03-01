
from abc import ABC, abstractmethod

class BaseFramework(ABC):

    def __init__(self,model,optimizer,criterion,inputs,ground_truths) -> None:
        self.model=model
        self.optim = optimizer
        self.criterion=criterion
        self.inputs=inputs 
        self.groud_truths=ground_truths
        pass

    def forward(self):
        pass

    def calculate_loss(self):
        pass

    def train(self):
        pass

    def test(self):
        pass 