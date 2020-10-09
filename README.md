HIDRA is a state-of-the-art deep-learning model for sea level forecasting based on temporal atmospheric and sea level data.

![Example sea level predictions (compared with NEMO).](images/example.png)

## Setup

**Requires**: Python ≥ 3.6  
Clone the repository, then use `pip` to install HIDRA in the active Python enviroment.
```bash
git clone https://github.com/lojzezust/HIDRA.git
pip install HIDRA
```

In you want to make changes to the HIDRA codebase, install the package in develop mode.
```bash
pip install -e HIDRA
```

## Running examples

A pretrained HIDRA model trained on Koper gauge station data is included.  We provide a [notebook example](examples/prediction.ipynb), showing how to use HIDRA for sea level forecasting on sample data.
