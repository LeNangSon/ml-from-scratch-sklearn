#%%
from abc import ABC, abstractmethod
import numpy as np


class BaseLoss(ABC):
    @abstractmethod
    def loss(self, y_true, y_pred):
        pass
    @abstractmethod
    def gradient(self, y_true, y_pred):
        pass
    def __call__(self, y_true, y_pred):
        return self.loss(y_true, y_pred)
    def loss_and_gradient(self,y_true,y_pred):
        loss_value = self.loss(y_true,y_pred)
        gradient_value = self.gradient(y_true,y_pred)
        return (loss_value, gradient_value)
    @staticmethod
    def _validate_inputs(y_true, y_pred):
        y_true = np.array(y_true, dtype = np.float64)
        y_pred = np.array(y_pred, dtype = np.float64)

        if y_true.shape != y_pred.shape:
            raise ValueError(
                "y_true and y_pred must have the same shape."
            )

        if y_true.size == 0 or y_pred.size == 0:
            raise ValueError(
                "Inputs must not be empty."
            )

        return y_true, y_pred
#%%
class HalfSquaredError(BaseLoss):
    def loss(self, y_true, y_pred):
        y_true, y_pred = self._validate_inputs(y_true,y_pred)
        n_samples = y_true.shape[0]
        error = y_true - y_pred
        return 1/(2 * n_samples) * np.sum(error ** 2)
    def gradient(self, y_true, y_pred):
        y_true, y_pred = self._validate_inputs(y_true,y_pred)
        n_samples = y_true.shape[0]
        return 1/n_samples * (y_pred - y_true)
#%%
class AbsoluteError(BaseLoss):
    def loss(self,y_true, y_pred):
        y_true, y_pred = self._validate_inputs(y_true, y_pred)
        n_samples = y_true.shape[0]
        return 1/n_samples * np.sum(np.abs(y_true - y_pred))
    def gradient(self,y_true,y_pred):
        y_true, y_pred = self._validate_inputs(y_true, y_pred)
        n_samples = y_true.shape[0]
        return 1/n_samples * np.sign(y_pred - y_true)
#%%
y_true = np.array([1.0, 2.0, 3.0])
y_pred = np.array([2.0, 2.0, 4.0])

loss_function = AbsoluteError()

loss_value = loss_function(y_true, y_pred)
gradient_value = loss_function.gradient(y_true, y_pred)
print(loss_value)
print(gradient_value)