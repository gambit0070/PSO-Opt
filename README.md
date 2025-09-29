# CITS4404  Project

## 📘 Description

This project implements and evaluates multiple swarm intelligence optimization algorithms (GWO, PSO, WOA) applied to technical trading strategies like MACD and SMA.  
It includes models, evaluators, utilities, and test scripts.

---

## 🧠 Optimizers

- **GWO** — *Grey Wolf Optimizer*
- **PSO** — *Particle Swarm Optimization*
- **WOA** — *Whale Optimization Algorithm*

> 💡 Each optimizer uses a `Run` function as the main entry point.

---

## 📈 MACD Bot

### 🔧 Model

- `MACD`: The core MACD trading model
- `signal_crossover`: Generates **buy/sell signal arrays**

### 🧪 Evaluation

- `MACD_Evaluator`: Evaluates MACD performance using a **fitness function**

---

## 📉 SMA Bot

- `SMA_bot`: Combines the **SMA trading model** and its **evaluator**

---

## 📊 Utilities

- `dataPro`: Loads data for **plotting only**
- `check_fitness`: Visualizes and checks **fitness values**

---

## 🧪 Test & Training Scripts

- `test_macd`: Trains and tests the **MACD model**, then visualizes results
- `test_woa`: **Unit test** for the WOA algorithm
- `train_woa_sma`: Trains **WOA** using the **SMA model**

## Instructions

Call the "run" function in test_macd.py to train and test
WOA and GWO for the data.
Modify the parameters setting in run function.
