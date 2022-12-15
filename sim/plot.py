import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context("paper", font_scale = 1.2)
color = sns.color_palette("tab10", 10)[0]


def plot_simulation(metrics):
    print(" > Exporting plots...")
    print(" > Plot 1: What are the different simulations (demand curve)? Which of them are feasible?")
    metrics['total_costs'] /= 1000000
    sns.scatterplot(metrics, x = "total_costs", y = "total_use", hue = "feasible")
    plt.title('All scenarios')
    plt.xlabel('Total amount spent on water by population (in million USD)')
    plt.ylabel('Total amount used by population (in acre-feet)')
    plt.savefig('output/scenarios/1-all_scenarios.png',dpi=600)
    plt.clf()


    print(" > Plot 2a: What is our menu of options? (equity vs. economic damage)")
    metrics_of_feasible = metrics.loc[metrics['feasible']==True]
    sns.scatterplot(metrics_of_feasible, x='total_costs', y='perc_spent_by_bottom_quintile', hue='total_use')
    plt.savefig('output/scenarios/2a-equity-vs-economy.png',dpi=600)
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
    metrics_of_feasible.loc[:,'in_frontier'] = in_frontier
    sns.scatterplot(metrics_of_feasible, x='total_costs', y='perc_spent_by_bottom_quintile', hue='in_frontier')
    plt.title('Comparison 1')
    plt.xlabel('Total cost for the population/economy')
    plt.ylabel('Percentage of total water cost paid by bottom quintile')
    plt.savefig('output/scenarios/2b-frontier.png',dpi=600)
    plt.clf()

    print(" > Plot 3: Of all values in our efficient frontier, what are the prices for each?")
    metrics_in_frontier = metrics_of_feasible.loc[metrics_of_feasible['in_frontier']==1,:].copy()

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
    plt.savefig('output/scenarios/per_gallon_price_bottom_quintile.png',dpi=600)
    plt.clf()


    sns.scatterplot(metrics, x='total_costs', y='prop_bottom_used_compared_to_top', hue='total_use')
    plt.xlabel('Total cost for the population/economy')
    plt.ylabel('Proportion of water usage by bottom quintile compared to top quintile')
    plt.savefig('output/scenarios/prop_bottom_used_compared_to_top.png',dpi=600)
    plt.clf()
    
    
    sns.scatterplot(metrics, x='total_costs', y='prop_per_gallon_price', hue='total_use')
    plt.xlabel('Total cost for the population/economy')
    plt.ylabel('Proportion of avg. water price for bottom vs. top quintile')
    plt.savefig('output/scenarios/prop_per_gallon_price.png',dpi=600)
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