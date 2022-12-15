from sim import econ
import pandas as pd

def simulation(price_1_list, price_2_list, thresholds, population_size, total_available):
    # metrics
    total_costs = []            # total money spent on water by everyone
    total_proportion_used = []  # percentage of water used at lower price
    total_water_use_by_bottom_quintile = [] 
    total_cost_for_bottom_quintile = []
    proportion_of_water_used_by_bottom_quintile = []
    variance_of_water_use = []  # proxy for inequality
    proportion_of_lower_price_water_used_by_bottom_quintile = []
    per_gallon_price_bottom_quintile = []
    per_gallon_price_top_quintile = []
    total_use_top_quintile = []
    total_use_bottom_quintile = []
    total_use = []
    feasible = [] 
    p1 = []
    p2 = []
    t = []

    print(" > Starting simulation...")
    i = 0
    for price1 in price_1_list:
        print(" > Simulation {perc}% done.".format(perc=round(100*i/len(price_1_list))))
        i += 1
        for price2 in price_2_list:
            for th in thresholds:
                if price2 >= price1:
                    population_df = econ.generate_scenario(population_size)
                    outcome = econ.simulate_demand(population_df, price1,price2, th)
                    
                
                    feasible.append(outcome["total_used"].sum() <= total_available)
                    total_proportion_used.append(outcome["used_at_lower_price"].sum()/outcome["total_used"].sum())
                    total_costs.append(outcome["total_spent"].sum())
                    total_use.append(outcome["total_used"].sum())
                    total_use_top_quintile.append(outcome.loc[outcome['income_quintile'] == 5,"total_used"].sum())
                    total_use_bottom_quintile.append(outcome.loc[outcome['income_quintile'] == 1,"total_used"].sum())
                    total_water_use_by_bottom_quintile.append(outcome.loc[outcome['income_quintile'] == 1, "total_used"].sum())
                    total_cost_for_bottom_quintile.append(outcome.loc[outcome['income_quintile'] == 1, "total_spent"].sum())
                    proportion_of_water_used_by_bottom_quintile.append(outcome.loc[outcome['income_quintile'] == 1, "total_used"].sum()/outcome["total_used"].sum())
                    proportion_of_lower_price_water_used_by_bottom_quintile.append(outcome.loc[outcome['income_quintile'] == 1, "total_used"].sum()/(0.001+outcome["used_at_lower_price"].sum()))
                    variance_of_water_use.append(outcome['total_used'].var())
                    per_gallon_price_bottom_quintile.append(outcome.loc[outcome['income_quintile'] == 1, "total_spent"].sum()/outcome.loc[outcome['income_quintile'] == 1, "total_used"].sum())
                    per_gallon_price_top_quintile.append(outcome.loc[outcome['income_quintile'] == 5, "total_spent"].sum()/outcome.loc[outcome['income_quintile'] == 1, "total_used"].sum())


                    # information about the scenario
                    p1.append(price1)
                    p2.append(price2)
                    t.append(th)
                    
    print(" > Simulation 100% done.")

    print(" > Saving metrics...")
    metrics = pd.DataFrame({
        'feasible': feasible,
        'total_proportion_used': total_proportion_used,
        'total_costs': total_costs,
        'total_use': total_use,
        'total_use_top_quintile': total_use_top_quintile,
        'total_use_bottom_quintile': total_use_bottom_quintile,
        'total_water_use_by_bottom_quintile': total_water_use_by_bottom_quintile,
        'total_cost_for_bottom_quintile': total_cost_for_bottom_quintile,
        'proportion_of_water_used_by_bottom_quintile': proportion_of_water_used_by_bottom_quintile,
        'proportion_of_lower_price_water_used_by_bottom_quintile': proportion_of_lower_price_water_used_by_bottom_quintile,
        'variance_of_water_use': variance_of_water_use,
        'per_gallon_price_bottom_quintile': per_gallon_price_bottom_quintile,
        'per_gallon_price_top_quintile': per_gallon_price_top_quintile,
        'price1': p1,
        'price2': p2,
        'threshold': t
    })              
    metrics['perc_spent_by_bottom_quintile'] = metrics['total_cost_for_bottom_quintile'] / metrics['total_costs']
    metrics['prop_bottom_used_compared_to_top'] = metrics['total_use_bottom_quintile'] / metrics['total_use_top_quintile']
    metrics['prop_per_gallon_price'] = metrics['per_gallon_price_bottom_quintile'] / metrics['per_gallon_price_top_quintile']

    return metrics
