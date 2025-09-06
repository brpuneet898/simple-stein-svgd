import numpy as np
from simple_stein_svgd import SVGD

# Target distribution: standard Gaussian
def log_prob(x):
    return -0.5 * np.sum(x**2, axis=1)

# Initialize particles
particles = np.random.randn(100, 1)

# Run SVGD
svgd = SVGD(log_prob)
for _ in range(100):
    particles = svgd.update(particles, stepsize=0.05)

print("Final particle mean:", np.mean(particles))
print("Final particle variance:", np.var(particles))
