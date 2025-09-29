import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PARAMETERS:
    # P: Price series (numpy array)
    ## High Frequency -
        # w1, w2, w3: Weights for SMA, LMA, EMA
        # d1, d2, d3: Durations (window sizes) for SMA, LMA, EMA
        # alpha3: EMA decay rate
    ## Low Frequency -
        # w4, w5, w6: Weights for SMA, LMA, EMA
        # d4, d5, d6: Durations (window sizes) for SMA, LMA, EMA
        # alpha6: EMA decay rate
    ## Smoothing -
        # w7, w8, w9: Weights for SMA, LMA, EMA
        # d7, d8, d9: Durations (window sizes) for SMA, LMA, EMA
        # alpha9: EMA decay rate

# USE GUIDE:
    ## Create an instance of the MACD class with the price series and parameters
        # params = np.array([w1, w2, w3, d1, d2, d3, alpha3, w4, w5, w6, d4, d5, d6, alpha6, w7, w8, w9, d7, d8, d9, alpha9])
        # macd_bot = MACD(P, params)
    ## Call the lines method to get the MACD line, Signal line, difference line and indication (buy [1], sell [-1], or neither [0] signals)
        # macd_line, signal_line, difference_line, indication = macd_bot.signal_crossover()
    ## Call the plot method to visualize the MACD and Signal lines, and obtain the lines and crossover signals (indication)
        # macd_line, signal_line, difference_line, indication = macd_bot.plot()
        ## Note: 
            # the P may have been trimmed (at the front) to match the length of the MACD and signal lines
            # OR the MACD and signal lines may have been trimmed to match the length of the P (ignore padding)



# Weighted sum of three components {SMA, LMA, EMA} 
## high frequency, low frequency, and smoothing components have the same function
class weighted_sum:
    def __init__(self, P, w_s, w_l, w_e, d_s, d_l, d_e, alpha):
        self.P = P
        self.w_s = w_s
        self.w_l = w_l
        self.w_e = w_e
        self.d_s = d_s
        self.d_l = d_l
        self.d_e = d_e
        self.alpha = alpha
    
    # add padding to the beginning of the price series
    # to account for the window size N which may differ between SMA, LMA, and EMA
    def pad(self, N):
        padding = -np.flip(self.P[1:N] - self.P[0]) + self.P[0]
        return np.append(padding, self.P)

    def sma(self):
        padded = self.pad(self.d_s)
        kernel = np.ones(self.d_s) / self.d_s
        return np.convolve(padded, kernel, 'valid')

    def lma(self):
        padded = self.pad(self.d_l)
        kernel = (2 / (self.d_l + 1)) * (1 - np.arange(self.d_l) / self.d_l)
        return np.convolve(padded, kernel, 'valid')

    def ema(self):
        padded = self.pad(self.d_e)
        kernel = (self.alpha) * ((1 - self.alpha) ** np.arange(self.d_e))
        return np.convolve(padded, kernel, mode='valid')

    def signal(self):
        # Calculate each component
        sma_part = self.sma()
        lma_part = self.lma()
        ema_part = self.ema()
    
        # Adjust lengths by trimming to match the shortest
        ## trim from the beginning which contains padding
        min_len = min(len(sma_part), len(lma_part), len(ema_part))
        sma_part = sma_part[-min_len:]
        lma_part = lma_part[-min_len:]
        ema_part = ema_part[-min_len:]
    
        # Weighted sum
        numerator = self.w_s * sma_part + self.w_l * lma_part + self.w_e * ema_part
        denominator = self.w_s + self.w_l + self.w_e

        # Avoid division by zero
        if denominator == 0:
            raise ValueError("Denominator in weighted sum is zero. Check weights.")
        
        return numerator / denominator
    


# MACD class
## Calls the weighted_sum class to calculate the MACD line and signal line
class MACD: 
    # def __init__(self, P, ws):
    #     self.init(P, ws[0], ws[1], ws[2],ws[3],ws[4],ws[5],ws[6],ws[7],ws[8],ws[9],ws[10],ws[11],ws[12],ws[13],ws[14],ws[15],ws[16],ws[17],ws[18],ws[19],ws[20])

    def __init__(self, P,w1, w2, w3, d1, d2, d3, alpha3, w4, w5, w6, d4, d5, d6, alpha6, w7, w8, w9, d7, d8, d9, alpha9):
        # Validate price series (P) 
        if not isinstance(P, (list, np.ndarray)):
            raise ValueError("Price data (P) must be a list or numpy array.")
        for p in P:
            if not isinstance(p, (int, float)):
                raise ValueError("Price data (P) must contain numeric values.")
            
        # Validate window sizes
        if len(P) < max(d1, d2, d3, d4, d5, d6, d7, d8, d9):
            raise ValueError("Price data (P) must be longer than the maximum window size.") 
        for d in [d1, d2, d3, d4, d5, d6, d7, d8, d9]:
            if not isinstance(d, int) or d < 0:
                raise ValueError("Window sizes (d1, d2, d3, d4, d5, d6) must be non-zero positive integers.")
        # if min(d4, d5) < max(d1, d2):
        #     raise ValueError("Low frequency window sizes (d4, d5, d6) must be greater than the high frequency windoe sizes (d1, d2, d3).")
        # Validate weights
        for w in [w1, w2, w3, w4, w5, w6, w7, w8, w9]:
            if not isinstance(w, (int, float)):
                raise ValueError("Weights (w1, w2, w3, w4, w5, w6, w7, w8, w9) must be numeric values.")
        # Validate decay rates // alpha
        for alpha in [alpha3, alpha6, alpha9]:
            if not isinstance(alpha, (int, float)):
                raise ValueError("Decay rates (alpha3, alpha6, alpha9) must be numeric values.")
        
        self.P = P
        # High frequency sum coefficients
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.alpha3 = alpha3
        # Low frequency sum coefficients
        self.w4 = w4
        self.w5 = w5
        self.w6 = w6
        self.d4 = d4
        self.d5 = d5 
        self.d6 = d6
        self.alpha6 = alpha6
        # Smoothing sum coefficients
        self.w7 = w7
        self.w8 = w8
        self.w9 = w9
        self.d7 = d7
        self.d8 = d8 
        self.d9 = d9
        self.alpha9 = alpha9
        
    def macd_line(self):
        # High frequency components
        high = weighted_sum(self.P, self.w1, self.w2, self.w3, self.d1, self.d2, self.d3, self.alpha3)
        high_freq = high.signal()
        # Low frequency components
        low  = weighted_sum(self.P, self.w4, self.w5, self.w6, self.d4, self.d5, self.d6, self.alpha6)
        low_freq = low.signal()

        # Ensure both arrays have same length
        min_len = min(len(high_freq), len(low_freq))
        high_freq = high_freq[-min_len:]
        low_freq = low_freq[-min_len:]

        # Calculate MACD line (difference between high and low frequency components)
        return high_freq - low_freq

    def lines(self): 
        # Calculate MACD line
        macd_line = self.macd_line()

        # Calculate Signal line (smoothing of MACD line)
        signal_sum = weighted_sum(macd_line, self.w7, self.w8, self.w9, self.d7, self.d8, self.d9, self.alpha9)  
        signal_line = signal_sum.signal()

        # Return both MACD and Signal lines
        return macd_line, signal_line
    
    def signal_crossover(self):
        macd_line, signal_line = self.lines()

        # Ensure both arrays have same length
        min_len = min(len(macd_line), len(signal_line))
        macd_line = macd_line[-min_len:]
        signal_line = signal_line[-min_len:]


        # Calculate difference between MACD and Signal lines
        difference_line = macd_line - signal_line 

        # Convolve with kernel from Equation (6)
        kernel = np.array([1, -1]) * 0.5
        indication = np.convolve(np.sign(difference_line), kernel, mode='valid')
        # Pad the start to match the input length
        indication = [0] + indication

        return macd_line, signal_line, difference_line, indication
    
    ## Can add a function to identify zero crossover points for the MACD line itself
        ## provide evidence of a change in the direction of a trend 
        ## but less confirmation of its momentum than a signal line crossover
    #def zero_crossover(self):
    #    zero_cross = []
    #    return zero_cross
    
    def plot(self):
        macd_line, signal_line, difference_line, indication = self.signal_crossover()
        P = self.P

        if len(P) > len(macd_line):
            # Adjust the length of the price series to match the MACD line
            P = P[-len(macd_line):]
        elif len(P) < len(macd_line):
            # Adjust the length of the arrays to match the price series
            indication = indication[-len(P):]
            macd_line = macd_line[-len(P):]
            signal_line = signal_line[-len(P):]
            difference_line = difference_line[-len(P):]

        # Plotting
        plt.figure(figsize=(12, 6))
        
        # Plot price series
        plt.plot(P, label='Price (P)', color='black', linestyle='-', marker='+', linewidth=1)
        # Plot MACD and Signal lines    
        plt.plot(macd_line, label='MACD Line', color='blue')
        plt.plot(signal_line, label='Signal Line', color='red')
        plt.plot(difference_line, label='Difference Line', color='lightgrey', linestyle='--')
        
        # Add buy/sell labels to legend and plot crossover points 
        plt.plot([], [], marker='^', color='green', linestyle='None', label='Buy Signal')
        plt.plot([], [], marker='v', color='red', linestyle='None', label='Sell Signal')
        for index, signal in enumerate(indication):
            if signal > 0:
                plt.plot(index, P[index], marker='^', color='green', markersize=10)
            elif signal < 0:
                plt.plot(index, P[index], marker='v', color='red', markersize=10)


        # Add horizontal line at y=0 for reference
        plt.axhline(y=0, color='lightgray', linewidth=1)
        plt.title('MACD Crossover Signals with Price')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.tight_layout()            
        plt.show()

        return macd_line, signal_line, difference_line, indication
    
class MACD_Evaluator:
    def __init__(self):
        self.close_data = []
        self.one_minus_fee_rate = 0.97
        self.numBuy = 0
        self.numSell = 0
        self.totalUsd = 0
        return
    
    def Dim(self):
        return 21
    
    def ReadData(self, path):
        df = pd.read_csv(path)
        self.close_data = np.array(df['close'])

    def Fitness(self, X, timeScale):
        d1 = int(np.round(X[3]))
        d2 = int(np.round(X[4]))
        d3 = int(np.round(X[5]))
        d4 = int(np.round(X[10]))
        d5 = int(np.round(X[11]))
        d6 = int(np.round(X[12]))
        d7 = int(np.round(X[17]))
        d8 = int(np.round(X[18]))
        d9 = int(np.round(X[19]))

        averageUsd = 0
        totalEarn = 0
        numLoop = len(self.close_data)//timeScale

        for k in range(numLoop):
          usd = 1000.0
          bitcoin = 0.0
          close_data = self.close_data[k * timeScale : k * timeScale + timeScale]
          
          macd = MACD(close_data, X[0],X[1],X[2],d1,d2,d3,X[6],X[7],X[8],X[9],d4,d5,d6, X[13],X[14],X[15], X[16], d7,d8,d9, X[20])
          _, _, _, result = macd.signal_crossover()
          for i in range(1, len(result)):
              d = close_data[i]
              #buy
              if result[i] > 0.5:
                  bitcoin = bitcoin + usd * self.one_minus_fee_rate / d
                  usd = 0.0
                  self.numBuy = self.numBuy + 1.0
              #sell
              elif result[i] < -0.5:
                  usd = usd + bitcoin * d * self.one_minus_fee_rate
                  bitcoin = 0.0
                  self.numSell = self.numSell + 1.0
        
          if bitcoin > 0:
              usd = usd +  bitcoin * close_data[-1] * self.one_minus_fee_rate
        
          totalEarn += (usd - 1000.0)

        averageUsd = totalEarn/numLoop
        self.totalUsd = totalEarn

        return averageUsd