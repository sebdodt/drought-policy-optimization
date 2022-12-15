from ml.run_ml import run_ml
from sim.run_sim import run_sim

amount_available = run_ml()
run_sim(amount_available)