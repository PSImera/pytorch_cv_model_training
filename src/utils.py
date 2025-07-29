import torch
from torchvision.transforms import v2

def transform(x, norm=True):
    transform_list = [
        v2.ToImage(),
        v2.Grayscale(),
        v2.ToDtype(torch.float32, scale=True),
    ]
    if norm:
        transform_list.append(v2.Normalize(mean=(0.5,), std=(0.5,)))

    pipeline = v2.Compose(transform_list)
    return pipeline(x)

class EarlyStopping():
    def __init__(self, mode='min', patience=10, threshold=0.0001, threshold_mode='rel'):
        if mode not in {'min', 'max'}:
            raise ValueError (f"modecan be min or max")
        if not isinstance (patience, int):
            raise TypeError(f"patience should ne int")
        if not isinstance (threshold, float):
            raise TypeError(f"threshold should be float")
        if threshold >= 1.0:
            raise ValueError("threshold should be >= 1.0")
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError("threshold_mode should be rel or ads")
        
        self.mode = mode
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.count = 0
        self.best = None

    def __call__(self, tracked_parameter):
        current = float(tracked_parameter)
        if self.best is None:
            self.best = current
            return False
        if self.changed_better(current, self.best):
            self.best = current
            self.count = 0
        else:
            self.count += 1

        if self.count >= self.patience:
            self.count = 0
            return True
        return False
    
    def changed_better(self, current, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            return current < best - best * self.threshold
        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return current < best - self.threshold
        elif self.mode == 'max' and self.threshold_mode == 'rel':
            return current > best + best * self.threshold
        else: # mode == 'max' and threshold_mode == 'abs
            return current > best + self.threshold
