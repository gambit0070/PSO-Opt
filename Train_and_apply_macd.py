from MACD_bot import *
from woa import *
from gwo import *
from pso import *
import time
import pandas as pd
import os

def read_data(path):
    df = pd.read_csv(path)
    
    # Make sure the date column is of type datetime
    df['date'] = pd.to_datetime(df['date'])

    # Split data by time
    df_before_2020 = df[df['date'] < '2020-01-01']
    df_after_2020 = df[df['date'] >= '2020-01-01']

    # Save close column as numpy array
    close_data_before_2020 = np.array(df_before_2020['close'])
    close_data_after_2020 = np.array(df_after_2020['close'])

    close_data_before_2020 = np.flip(close_data_before_2020)
    close_data_after_2020 = np.flip(close_data_after_2020)
    return close_data_before_2020, close_data_after_2020

#train on WOA and GWO
def train_models(bot, timeScale,mins,maxs,randSeed,populationNumber, max_iter):
    # Test WOA
    woa = WOA_Optimizer(
        fitness = bot.Fitness,
        seed =  randSeed, 
        dim = bot.Dim(), 
        Xmin = mins, 
        Xmax = maxs, 
        populationNumber = populationNumber, 
        max_iter = max_iter, 
        scale = timeScale
    )
    woa_target, woa_fit, woa_fits = woa.Run()
    print("WOA Training Complete")

    # Test GWO
    gwo = GreyWolfOptimiser(
        fitness_function = bot.Fitness,
        max_iter = max_iter,
        num_wolves = populationNumber,
        dim = bot.Dim(),
        minx = mins,
        maxx = maxs,
        seed = randSeed,
        fitness_args = (timeScale,)
    )
    gwo_target, gwo_fit = gwo.run()
    print("GWO Training Complete")

    return woa_target, woa_fit, gwo_target, gwo_fit

#work on testing data
def apply_models(bot, woa_target, gwo_target, timeScale):
    woa_fit = bot.Fitness(woa_target, timeScale)
    woa_total = bot.totalUsd
    gwo_fit = bot.Fitness(gwo_target, timeScale)
    gwo_total = bot.totalUsd
    return woa_total, gwo_total

def run():
    #train_data, test_data = read_data("btc_hourly.csv")
    train_data, test_data = read_data("BTC-Daily.csv")

    bot = MACD_Evaluator()
    bot.close_data = train_data

    #set parameters
    timeScale = 200
    temp = [0.1, 0.1, 0.1, 1,1,1, 0.1]
    mins = temp + temp + temp 
    max_temp = [1.0, 1.0, 1.0, 50, 50 ,50 , 1.0]
    maxs = max_temp + max_temp + max_temp
    randSeed = int(time.time())
    populationNumber = 100
    max_iter = 50

    woa_target, woa_fit, gwo_target, gwo_fit = train_models(bot,timeScale,mins,maxs,randSeed,populationNumber, max_iter)

    bot.close_data = test_data
    woa_total, gwo_total  = apply_models(bot, woa_target, gwo_target, len(test_data))

    # --- Save results to CSV ---
    results = {
        "Bot": ["MACD", "MACD"],
        "Optimizer": ["WOA", "GWO"],
        "Train Fitness": [woa_fit, gwo_fit],
        "Test USD": [woa_total, gwo_total],
        "Optimal Parameters": [str([f"{p:.4f}" for p in woa_target]), str([f"{p:.4f}" for p in gwo_target])]
    }

    df = pd.DataFrame(results)
    results_file = "macd_results.csv"
    if os.path.exists(results_file):
        print(f"Appending results to {results_file}")
        existing = pd.read_csv(results_file)
        df = pd.concat([existing, df], ignore_index=True)
    else:
        print(f"Creating new results file: {results_file}")
    df.to_csv(results_file, index=False)

if __name__ == "__main__":            
    run()


