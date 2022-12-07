import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context("paper", font_scale = 1.2)
color = sns.color_palette("tab10", 10)[0]

def generate_scenario(population_size):
    '''
    Sets parameters according to our economic analysis and literature review.
    Creates a dataframe of individuals
    '''
    # print(" > Generating scenario...")
    population = population_size    # size of our town
    household_size = 4              # number of people per household
    avg_demand_elasticity = -0.435  # in $/acre-feet of water
    avg_use = 100                   # in gallons per day per person
    avg_water_cost = 77             # in $ per month per household of 4
    median_income = 78672           # for population
    sigma_income = 0.73796
    
    
    # transform parameters
    yearly_use = avg_use * 365.25 * household_size          # yearly use in gallons per household
    yearly_use_af = yearly_use / 325851                     # yearly use in acre-feet per household
    yearly_cost = avg_water_cost * 12 * household_size/ 4   # yearly cost in $ per household
    current_cost_per_af = yearly_cost/yearly_use_af

    # other assumptions
    part_of_use_that_is_fixed = 2/3 # https://www.google.com/search?q=water+usage+by+income&client=safari&rls=en&sxsrf=ALiCzsbcJZmmS70aTZQlw8zcg9E0IwfbUA:1670270330064&source=lnms&tbm=isch&sa=X&ved=2ahUKEwiI7M7toeP7AhVXmHIEHRW1CE8Q_AUoAXoECAEQAw&biw=1440&bih=812&dpr=2#imgrc=g1km3zfEH6yM7M

    # generate individuals
    income = np.random.lognormal(mean=np.log(median_income), sigma=sigma_income, size=population)    # https://statisticsbyjim.com/probability/global-income-distributions/
    income_quintile = pd.qcut(income,5, labels=range(1,6))
    current_use =  part_of_use_that_is_fixed * yearly_use_af + (((1-part_of_use_that_is_fixed)*yearly_use_af)/(median_income)) * income   # https://www.researchgate.net/figure/Monthly-household-income-vs-daily-water-consumption_fig3_325584808                                                     # https://en.wikipedia.org/wiki/Residential_water_use_in_the_U.S._and_Canada
    current_use += np.random.normal(0,current_use/10)
    current_cost = current_cost_per_af * current_use
    elasticity = avg_demand_elasticity + ((np.log(income)-np.log(median_income))/-10) # https://www.researchgate.net/figure/Estimated-price-elasticity-of-water-demand-for-low-middle-and-high-income-groups_tbl1_228917285
    
    population_df = pd.DataFrame({
        'income':income,
        'income_quintile':income_quintile,
        'current_use':current_use,
        'current_cost':current_cost,
        'elasticity':elasticity
    })
    return population_df
    


def simulate_demand(population_df, price1, price2, threshold):
    '''
    Input: price (in $ per acre-foot)
    Output: water usage (in acre-foot per year)
    '''
    # print(" > Simulating demand...")
    price_below_threshold = min(price1,price2)
    price_above_threshold = max(price1,price2)

    ## demand for lower price 
    population_df['theoretical_demand_lower_price'] = population_df['current_use'] * (price_below_threshold/population_df['current_cost']) ** population_df['elasticity']
    population_df['theoretical_demand_higher_price'] = population_df['current_use'] * (price_above_threshold/population_df['current_cost']) ** population_df['elasticity']
    population_df['threshold'] = threshold
    population_df['used_at_lower_price'] = population_df[['theoretical_demand_lower_price','threshold']].min(axis=1)
    
    ## demand for higher price
    population_df.loc[population_df['theoretical_demand_lower_price']<threshold,'used_at_higher_price'] = 0
    population_df.loc[population_df['theoretical_demand_higher_price']<threshold,'used_at_higher_price'] = 0
    population_df.loc[population_df['theoretical_demand_higher_price']>=threshold,'used_at_higher_price'] = population_df['theoretical_demand_higher_price'] - threshold

    ## total expenses
    population_df['total_used'] = population_df['used_at_lower_price'] + population_df['used_at_higher_price'] 
    population_df['total_spent'] = population_df['used_at_lower_price'] * price_below_threshold + population_df['used_at_higher_price'] * price_above_threshold
    return population_df.drop('threshold',axis=1)


#def full_simulation(population_df, price1_list, price2_list, thresholds):
    


if __name__=='__main__':
    population_size = 5000
    total_available = 0.4124291510893835 * population_size # given by Seb's ML pipeline
    
    thresholds = np.arange(0,10,0.5)
    price_1_list = range(1000, 4000,100)
    price_2_list = range(1000, 4000,100)

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
                    population_df = generate_scenario(population_size)
                    outcome = simulate_demand(population_df, price1,price2, th)
                    
                
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


    print(" > Exporting plots...")
    print(" > Plot 1: What are the different simulations (demand curve)? Which of them are feasible?")
    metrics['total_costs'] /= 1000000
    sns.scatterplot(metrics, x = "total_costs", y = "total_use", hue = "feasible")
    plt.title('All scenarios')
    plt.xlabel('Total amount spent on water by population (in million USD)')
    plt.ylabel('Total amount used by population (in acre-feet)')
    plt.savefig('1-all_scenarios.png',dpi=600)
    plt.clf()


    print(" > Plot 2a: What is our menu of options? (equity vs. economic damage)")
    metrics_of_feasible = metrics.loc[metrics['feasible']==True]
    sns.scatterplot(metrics_of_feasible, x='total_costs', y='perc_spent_by_bottom_quintile', hue='total_use')
    plt.savefig('2a-equity-vs-economy.png',dpi=600)
    plt.clf()

    print(" > Plot2b: Menu of options with efficient frontier")
    
    def is_in_frontier(df, metric1, metric2, row):
        df_filtered = df.loc[((df[metric1] < df.loc[row,metric1]) & (df[metric2] < df.loc[row, metric2])),:].copy()
        if len(df_filtered) == 0:
            return True
        else:
            return False
 
    in_frontier = []
    for i in metrics_of_feasible.index:
        in_frontier.append(is_in_frontier(metrics_of_feasible,'total_costs', 'perc_spent_by_bottom_quintile', i))
    metrics_of_feasible['in_frontier'] = in_frontier
    sns.scatterplot(metrics_of_feasible, x='total_costs', y='perc_spent_by_bottom_quintile', hue='in_frontier')
    plt.title('Comparison 1')
    plt.xlabel('Total cost for the population/economy')
    plt.ylabel('Percentage of total water cost paid by bottom quintile')
    plt.savefig('2b-frontier.png',dpi=600)
    plt.clf()

    print(" > Plot 3: Of all values in our efficient frontier, what are the prices for each?")
    metrics_in_frontier = metrics_of_feasible.loc[metrics_of_feasible['in_frontier']==1,:]

    # fig, axes = plt.subplots(3, 1, figsize=(9, 9))
    # sns.lineplot(ax=axes[0], data=metrics_in_frontier, x='total_costs', y='price1')
    # sns.lineplot(ax=axes[1], data=metrics_in_frontier, x='total_costs', y='price2')
    # sns.lineplot(ax=axes[2], data=metrics_in_frontier, x='total_costs', y='threshold')
    # plt.savefig('3-frontier-choices.png',dpi=600)
    # plt.clf()
    # print(metrics)





# fig.suptitle('Pokemon Stats by Generation')

# sns.boxplot(ax=axes[0, 0], data=pokemon, x='Generation', y='Attack')
# sns.boxplot(ax=axes[0, 1], data=pokemon, x='Generation', y='Defense')
# sns.boxplot(ax=axes[0, 2], data=pokemon, x='Generation', y='Speed')





    # sns.scatterplot(metrics, x = "total_costs", y = "variance_of_water_use")
    # plt.savefig("variance_as_function_total_costs.png")
    # plt.clf()

    
    #fig, (ax1, ax2) = plt.subplots(2,1, figsize = (10,8))
   # ax1.plot(thresholds, total_proportion_used)
   # ax1.set(xlabel = "Thresholds", ylabel = "Proportion of Lower Price Water Use", title = "Proportion of Lower Price Water Use as a Function of Threshold")

   # ax2.plot(thresholds, total_costs)
   # ax2.set(xlabel = "Thresholds", ylabel = "Total Cost of Water Use", title = "Total Cost of Water Use as a Function of Threshold")
    # fig.subplots_adjust(hspace = 0.3)
    # plt.show()
    

    # ## Sebs
    # metrics['perc_spent_by_bottom_quintile'] = metrics['total_cost_for_bottom_quintile'] / metrics['total_costs']
    sns.scatterplot(metrics, x='total_costs', y='per_gallon_price_bottom_quintile', hue='total_use')
    plt.xlabel('Total cost for the population/economy')
    plt.ylabel('Average water price for bottom quintile (in acre-feet/USD)')
    plt.savefig('per_gallon_price_bottom_quintile.png',dpi=600)
    plt.clf()


    sns.scatterplot(metrics, x='total_costs', y='prop_bottom_used_compared_to_top', hue='total_use')
    plt.xlabel('Total cost for the population/economy')
    plt.ylabel('Proportion of water usage by bottom quintile compared to top quintile')
    plt.savefig('prop_bottom_used_compared_to_top.png',dpi=600)
    plt.clf()
    
    
    sns.scatterplot(metrics, x='total_costs', y='prop_per_gallon_price', hue='total_use')
    plt.xlabel('Total cost for the population/economy')
    plt.ylabel('Proportion of avg. water price for bottom vs. top quintile')
    plt.savefig('prop_per_gallon_price.png',dpi=600)
    plt.clf()

    # sns.scatterplot(metrics, x='total_costs', y='total_proportion_used', hue='total_use')
    # plt.savefig('total_proportion_used.png')
    # plt.clf()

    # sns.scatterplot(metrics, x='total_costs', y='total_use', hue='total_use')
    # plt.savefig('total_use.png')
    # plt.clf()

    # sns.scatterplot(metrics, x='total_costs', y='total_water_use_by_bottom_quintile', hue='total_use')
    # plt.savefig('total_water_use_by_bottom_quintile.png')
    # plt.clf()

    # sns.scatterplot(metrics, x='total_costs', y='total_cost_for_bottom_quintile', hue='total_use')
    # plt.savefig('total_cost_for_bottom_quintile.png')
    # plt.clf()

    # sns.scatterplot(metrics, x='total_costs', y='proportion_of_water_used_by_bottom_quintile', hue='total_use')
    # plt.savefig('proportion_of_water_used_by_bottom_quintile.png')
    # plt.clf()

    # sns.scatterplot(metrics, x='total_costs', y='proportion_of_lower_price_water_used_by_bottom_quintile', hue='total_use')
    # plt.savefig('proportion_of_lower_price_water_used_by_bottom_quintile.png')
    # plt.clf()

    # sns.scatterplot(metrics, x='total_costs', y='variance_of_water_use', hue='total_use')
    # plt.savefig('variance_of_water_use.png')
    # plt.clf()
    

    
    print(" > Done.")
    
    #price1 = 2060.82     # per acre-foot (current price = 2060.82)
    #price2 = 4000        # per acre-foot
    #threshold = 0.5      # in acre-feet per year

   # population_df = generate_scenario(population_size)
    #population_df = simulate_demand(population_df, price1, price2, threshold)
    #print(""" > In this scenario, {a} acre-feet of water are available. {u} acre-feet were used.""".format(a=round(total_available), u=round(population_df['total_used'].sum())))
    
    #print(population_df["elasticity"]) # this table shows each individual in the town per row, and their water use and expenditures at our price level.
    #print(population_df["used_at_lower_price"].sum())
    # print(population_df["used_at_higher_price"].sum())
    # (`Current use` and `Current cost` refer to how much they are currently using -- before our intervention)