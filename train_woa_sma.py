from SMA_bot import *
from woa import *
from pso import *
from gwo import *
import numpy as np
import time

def draw(data , sma_short, sma_long, sma_diff):
    x = np.arange(len(data))
    plt.plot(x, data, color = 'black')
    plt.plot(x, sma_short, color='blue')
    plt.plot(x, sma_long, color ='red')
    plt.plot(x, sma_diff, color ='gray' , linestyle='--')

    plt.title("Float Array Plot")
    plt.xlabel("Index")
    plt.ylabel("Value")

    plt.show()

if __name__ == "__main__":
    bot = SMA_Evaluator()
    #bot.Read_test_data("btc_hourly.csv")
    bot.Read_test_data("BTC-Daily.csv") 
    cof = 1
    timeScale = 200
    #bot.close_data = [200 - i for i in range(100)] + [ i + 100 for i in range(200)] + [ 300 - i for i in range(100)]
    
    usd = bot.Fitness(np.array([10 * cof, 20 * cof]), timeScale)

    print("\n")
    print("final usd:", usd)
    print("num buy:" , bot.numBuy)
    print("num sell:", bot.numSell)

    #draw(bot.close_data, bot.sma_short, bot.sma_long, bot.signal * 100)

    woa = WOA_Optimizer(bot.Fitness, 42, 2, [5, 5]*cof, [25, 25]*cof, 500, 100, timeScale)
    X_target = woa.Run()
    fit = woa.Fitness(X_target, timeScale)
    print("best_result:", X_target)
    print("fitness:" , fit)

    #test Pso
    pso = PSO(bot.Fitness, int(time.time()), 2, [5, 5]*cof, [25, 25]*cof, 500, 100, timeScale)
    pso_target = pso.Run()
    pso_fit = pso.Fitness(pso_target, timeScale)
    print("pso_best_result:", pso_target)
    print("pso_fitness:" , pso_fit)

    #test Grey Wolf Optimizer
    gwo = GreyWolfOptimiser(
        fitness_function=bot.Fitness,
        max_iter=500,
        num_wolves=100,
        dim=2,
        minx=[5, 5],
        maxx=[25, 25],
        seed=42,
        fitness_args=(timeScale,)
    )
    gwo_target, gwo_fit = gwo.run()
    print("gwo_best_result:", gwo_target)
    print("gwo_fitness:", gwo_fit)

    print("end")