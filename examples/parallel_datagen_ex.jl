using Distributed
using JLSO
using QuasiMonteCarlo
addprocs(10)
@everywhere using RainMakerChallenge2025
@everywhere using ProgressMeter

n = 10
lb = [0, -2000, 0, -180, -90, 270, 270, -5, -5, 5]
ub = [2, 5000, 30, 180, 90, 300, 300, 5, 5, 50]

s = QuasiMonteCarlo.sample(n, lb, ub, LatinHypercubeSample())
max_precipitation(s[:,1])
sols = @showprogress pmap(sample -> (sample, max_precipitation(sample)),
                          eachcol(s))
sampled_params = reduce(hcat,first.(sols))
sampled_outputs = last.(sols)

#save the data
# JLSO.save("10data.jlso", Dict(:d=>(inputs = sampled_params, outputs = sampled_outputs)))