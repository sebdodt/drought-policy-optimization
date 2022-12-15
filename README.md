# drought-policy-optimization



## How to run

**Clone the repository**

First, navigate to the directory where the repository should be located

Then clone the repository
```
git clone git@github.com:sebdodt/drought-policy-optimization.git
```


**Set up environment**

Create a new virtual environment
```
conda create --name venv python=3.10.6
conda activate venv
```

Install dependencies
```
pip install -r requirements.txt
```


**Run the pipeline**

You can run the pipeline with the following command
```
python run.py
```


**Results**

The results of the machine learning training will be stored in `output/ml_performance`

The results from the simulation will be stored in `output/scenarios`