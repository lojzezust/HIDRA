# HIDRA 1.0: Deep-learning-based method for sea level forecasting

HIDRA is a state-of-the-art deep-learning model for sea level forecasting based on temporal atmospheric and sea level data.

![Example sea level predictions (compared with NEMO).](images/example.png)

## Setup

**Requires**: Python ≥ 3.6  
Clone the repository, then use `pip` to install HIDRA in the active Python enviroment.
```bash
git clone https://github.com/lojzezust/HIDRA.git
pip install HIDRA/
```

In you want to make changes to the HIDRA codebase, install the package in develop mode.
```bash
pip install -e HIDRA/
```

  
## Usage

HIDRA is implemented as a Tensorflow Keras model, which enables straight-forward training and inference.
```python
from hidra import HIDRA
model = HIDRA()

# Training
model.fit(...)

# Inference
model.predict(...)
```

A pretrained HIDRA model trained on Koper gauge station data is included in the repository. We provide a [notebook example](examples/prediction.ipynb), showing how to use HIDRA with pretrained weights for sea level forecasting on sample data.
