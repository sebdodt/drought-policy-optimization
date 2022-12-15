import pandas as pd
import numpy as np
from sim import econ, plot, simulation




def run_sim(total_available):
    population_size = 5000
    total_available *= population_size 
    thresholds = np.arange(0,10,0.5)
    price_1_list = range(1000, 4000,100)
    price_2_list = range(1000, 4000,100)

    metrics = simulation.simulation(price_1_list, price_2_list, thresholds, population_size, total_available)
    plot.plot_simulation(metrics)


if __name__=='__main__':
    run_sim(0.4124291510893835)