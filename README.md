# 🌧️ RainMaker Challenge 2025

*Make it rain! ☔*

A Julia package for the RainMaker Challenge where the goal is simple: **maximize rainfall in Pittsburgh, PA** by tuning atmospheric and geographical parameters.

## 🎯 The Challenge

Given 10 tunable parameters, find the combination that produces the most precipitation:
- 🏔️ Mountain height, size, and location
- 🌡️ Temperature (equator, poles, USA, Pennsylvania)
- 🌍 Orography scale
- 💨 Zonal wind speed

## 🧠 Approach

Build surrogates or use optimization techniques to efficiently explore the parameter space instead of running expensive climate simulations for every guess.

## 📁 What's Inside

```
├── src/
│   └── rainmaker.jl          # Core max_precipitation function
├── examples/
│   ├── parallel_datagen_ex.jl    # Generate training data in parallel
│   ├── surrogate_nn_ex.jl        # Neural network surrogate + optimization
│   └── surrogate_elm_ex.jl       # Extreme Learning Machine surrogate
├── data/
│   ├── 10kdata.jlso          # 10k parameter-precipitation pairs
│   └── 100kdata.jlso         # 100k parameter-precipitation pairs  
└── models/
    └── surrogate_nn_*.jlso   # Pre-trained neural network models
```

## 🚀 Quick Start

```julia
using RainMakerChallenge2025

# Try the default parameters
params = [1, 0, 1, -80, 40.45, 300, 273, 0, 0, 35]
precipitation = max_precipitation(params)
```

## 🎉 Current Best

The neural network surrogate found parameters yielding **285mm** predicted rainfall (actual: 210mm). Can you do better?

---
*Built with ❤️ using RainMaker.jl and SpeedyWeather.jl* 