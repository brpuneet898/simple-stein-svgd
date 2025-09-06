# simple-stein-svgd

A minimal Python package implementing Stein Variational Gradient Descent (SVGD).

## Installation (TestPyPI)
```bash
pip install -i https://test.pypi.org/simple/ simple-stein-svgd
```

## Example:

```python
import numpy as np
from simple_stein_svgd import SVGD

def log_prob(x):
    return -0.5 * np.sum(x**2, axis=1)

particles = np.random.randn(100, 1)
svgd = SVGD(log_prob)
for _ in range(100):
    particles = svgd.update(particles, stepsize=0.05)

print("Mean:", np.mean(particles))
print("Variance:", np.var(particles))
```