

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class SMA_Evaluator:
    def __init__(self):
        self.Reset()
    
    def Reset(self):
        #self.close_data = []
        #self.open_data = []

        self.sma_diff = []
        self.sma_short = []
        self.sma_long = []

        self.numBuy = 0
        self.numSell = 0

        self.bitcoin = 0
        self.usd = 1000.0
        self.one_minus_fee_rate = 0.97
        return 
    
    def Read_test_data(self, path):
        df = pd.read_csv(path)
        self.close_data = np.array(df['close'])
        self.open_data = np.array(df['open'])

    def Fitness(self , X, scale):
        self.Reset()
        x0 = int(np.round(X[0]))
        x1 = int(np.round(X[1]))
        #x2 = int(np.round(X[2])) #time_scale
        x2 = scale

        averageUsd = 0
        numLoop = len(self.close_data)//x2

        self.sma_short = self.wma(self.close_data, x0, self.sma_filter(x0))
        self.sma_long = self.wma(self.close_data, x1, self.sma_filter(x1))
        self.sma_diff = self.sma_short - self.sma_long
        sign_data = np.sign(self.sma_diff)
        self.signal = self.wma(sign_data, 2, [0.5, -0.5])
        for k in range(numLoop):
          usd = 1000.0
          bitcoin = 0.0
          close_data = self.close_data[k * x2 : k * x2 + x2]
          sma_short = self.wma(close_data, x0, self.sma_filter(x0))
          sma_long =  self.wma(close_data, x1, self.sma_filter(x1))
          sign_data = np.sign(sma_short - sma_long)
          result  = self.wma(sign_data, 2, [0.5, -0.5])
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
        
          averageUsd += usd
        averageUsd = averageUsd/numLoop
        return averageUsd
    
    def pad(self, P,N):
        #padding = -np.flip(P[1:N])
        padding = np.ones(N - 1) * P[0]
        return  np.append(padding, P)
    
    def sma_filter(self, N):
        return np.ones(N)/N
        
    def wma(self, P,N,kernel):
        return np.convolve(self.pad(P,N), kernel, 'valid')
        




