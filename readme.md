# EFDD (Enhanced Frequency Domain Decomposition)

This module implements the Enhanced Frequency Domain Decomposition (EFDD) method for structural health monitoring using acceleration data.

## Installation

```bash
pip install numpy scipy
```

## Usage 

```python
import numpy as np
from efdd import EFDD

# Example data
Acc = # Acceleration data (samples x channels)
fs = 1000  # Sampling frequency
Nc = 4  # Number of channels

# Initialize EFDD
efdd = EFDD(Acc, fs, Nc)

# Calculate PSD matrix
psd, freq = efdd.get_psd_matrix()

# Get eigenvalues and mode shapes
single_value, mode_shapes = efdd.get_eigen_values()

# Calculate MAC value between two mode shapes
mac_value = efdd.MacVal(mode_shapes[:, 0], mode_shapes[:, 1])

# Get mode shape for a specific frequency
mode_shape = efdd.get_modeshape(frequency=10.0)

print("Mode Shape at 10 Hz:", mode_shape)
```

