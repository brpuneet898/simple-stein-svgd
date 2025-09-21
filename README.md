# simple-stein-svgd

A minimal Python package implementing Stein Variational Gradient Descent (SVGD).

## Installation (TestPyPI)
```bash
!pip install --extra-index-url https://pypi.org/simple/ -i https://test.pypi.org/simple/ simple-stein-svgd
```

## Example:

```python
import numpy as np
from simple_stein_svgd import SVGD

def log_prob(x):
    return -0.5 * np.sum(x**2, axis=1)

particles = np.random.randn(100, 1)
svgd = SVGD(log_prob)

for t in range(100):
    res = svgd.update(particles, stepsize=0.05)
    particles = res.particles
    if (t + 1) % 10 == 0:
        print(f"iter {t+1:3d} | KSD={res.ksd:.5f} | wall-time={res.wall_time*1e3:.2f} ms")

print("Mean:", np.mean(particles))
print("Variance:", np.var(particles))
```
