using JLSO
using LinearAlgebra
using Statistics
using LinearSolve
using Distributions
using NNlib
using Random
using Optimization
using OptimizationOptimJL
using Zygote
using RainMakerChallenge2025

# Set BLAS threads for faster matrix operations
LinearAlgebra.BLAS.set_num_threads(32)
println("Using $(LinearAlgebra.BLAS.get_num_threads()) BLAS threads")

minmaxnorm(data, lb, ub, norm_min=0., norm_max=1.) = @. norm_min + (data - lb) * (norm_max - norm_min) / (ub - lb)
minmaxdenorm(data, lb, ub, norm_min=0., norm_max=1.) = @. lb + (data - norm_min) * (ub - lb) / (norm_max - norm_min)

path = joinpath(dirname(@__DIR__), "data", "10kdata.jlso")
data = JLSO.load(path)

input_data = data[:d][:inputs]
output_data = data[:d][:outputs]

inputs_lb = [0., -2000., 0., -180., -90., 270., 270., -5., -5., 5.]
inputs_ub = [2., 5000., 30., 180., 90., 300., 300., 5., 5., 50.]
input_data_norm = minmaxnorm(input_data, inputs_lb, inputs_ub)

output_data = reshape(output_data, (1,:))
outputs_lb, outputs_ub = extrema(output_data)
output_data_norm = minmaxnorm(output_data, outputs_lb, outputs_ub)

tsplit = 9000
input_data_norm_train = input_data_norm[:, 1:tsplit]
input_data_norm_valid = input_data_norm[:, tsplit+1:end]
output_data_norm_train = output_data_norm[:, 1:tsplit]
output_data_norm_valid = output_data_norm[:, tsplit+1:end]

mutable struct ELM
    W_in::Matrix{Float64}
    b::Vector{Float64}
    W_out::Matrix{Float64}
    n_hidden::Int
    activation::Function
end

function ELM(n_inputs::Int, n_hidden::Int, n_outputs::Int; activation=tanh, seed=42, w_init = randn, b_init = randn)
    Random.seed!(seed)
    W_in = w_init(n_hidden, n_inputs) 
    b = b_init(n_hidden)
    W_out = zeros(n_hidden, n_outputs)
    return ELM(W_in, b, W_out, n_hidden, activation)
end

function train_regularized!(elm::ELM, X::Matrix{Float64}, Y::Matrix{Float64}, lambda::Float64=1e-6)
    H = elm.activation.(elm.W_in * X .+ elm.b)
    
    # Use the normal equations but with efficient linear solve
    # (H*H' + λI) * W = H*Y'  =>  W = (H*H' + λI) \ (H*Y')
    A = H * H' + lambda * LinearAlgebra.I(size(H, 1))
    b = H * Y'
    
    # Flatten b to vector since we have single output
    b_vec = vec(b)
    
    prob = LinearProblem(A, b_vec)
    sol = solve(prob)
    
    # Reshape back to matrix form
    elm.W_out = reshape(sol.u, :, 1)
    return elm
end

function predict(elm::ELM, X::Matrix{Float64})
    H = elm.activation.(elm.W_in * X .+ elm.b)
    return elm.W_out' * H
end

n_inputs = size(input_data_norm_train, 1)
n_outputs = size(output_data_norm_train, 1)
n_hidden = 15000

println("Training ELM with $n_hidden hidden neurons...")
d = Uniform(-1.0 / sqrt(n_inputs), 1.0 / sqrt(n_inputs))
elm = ELM(n_inputs, n_hidden, n_outputs; activation=leakyrelu, w_init = (x,y)->rand(d,x,y), b_init = (x)->zeros(x))

train_regularized!(elm, input_data_norm_train, output_data_norm_train, 1e-8)

output_pred_norm_valid = predict(elm, input_data_norm_valid)
output_pred_norm_train = predict(elm, input_data_norm_train)

train_mse = mean((output_data_norm_train .- output_pred_norm_train).^2)
valid_mse = mean((output_data_norm_valid .- output_pred_norm_valid).^2)

println("Training MSE (normalized): $train_mse")
println("Validation MSE (normalized): $valid_mse")

output_pred_train = minmaxdenorm(output_pred_norm_train, outputs_lb, outputs_ub)
output_pred_valid = minmaxdenorm(output_pred_norm_valid, outputs_lb, outputs_ub)
output_train_actual = minmaxdenorm(output_data_norm_train, outputs_lb, outputs_ub)
output_valid_actual = minmaxdenorm(output_data_norm_valid, outputs_lb, outputs_ub)

train_mse_actual = mean((output_train_actual .- output_pred_train).^2)
valid_mse_actual = mean((output_valid_actual .- output_pred_valid).^2)

println("Training MSE (actual scale): $train_mse_actual")
println("Validation MSE (actual scale): $valid_mse_actual")



# Define surrogate function using the trained ELM
surrogate(x) = predict(elm, x)

# Optimization using SciML Optimization.jl - batch optimization like original
function objective_func(x, p)
    n_params = 10
    n_samples = 5
    
    # Reshape x to matrix form (10 parameters × 5 samples)
    X = reshape(x, n_params, n_samples)
    
    # Get predictions from surrogate for ALL samples
    y_pred = surrogate(X)  # This returns predictions for all 5 samples
    
    # Maximize total precipitation by minimizing negative sum of ALL predictions
    obj = -sum(y_pred)
    
    # Add penalty for bounds violations (x should be in [0,1])
    penalty = 0.0
    λ = 100.0
    
    for xi in x
        if xi < 0.0
            penalty += λ * xi^2
        elseif xi > 1.0
            penalty += λ * (xi - 1.0)^2
        end
    end
    
    return obj + penalty
end

# Set up optimization problem - now optimizing 50 variables (10 params × 5 samples)
x0 = rand(50)  # Initial guess for 10 parameters × 5 samples
lb = zeros(50)  # Lower bounds
ub = ones(50)   # Upper bounds

# Create optimization function with automatic differentiation
optf = Optimization.OptimizationFunction(objective_func, Optimization.AutoZygote())

# Create optimization problem with box constraints
prob = Optimization.OptimizationProblem(optf, x0, lb=lb, ub=ub)

# Solve using LBFGS (good for smooth problems) 
println("Starting optimization to find optimal parameters...")
sol = solve(prob, LBFGS(), maxiters=10000)

println("Optimization completed!")
println("Final objective value: $(sol.objective)")
println("Converged: $(sol.retcode)")

# Extract best parameters from the 5 optimized samples
X_optimal = reshape(sol.u, 10, 5)  # Reshape to (10 params × 5 samples)

# Get predictions for all 5 samples
pred_all_norm = surrogate(X_optimal)  # Vector of 5 predictions

# Find the best sample (highest precipitation)
best_idx = argmax(pred_all_norm)[2]
best_params_norm = X_optimal[:,best_idx]  # Extract best parameter set
best_params_actual = minmaxdenorm(reshape(best_params_norm, :, 1), inputs_lb, inputs_ub)[:, 1]

println("Evaluating best parameters with actual simulator...")
actual_precip = max_precipitation(best_params_actual)

# Get predicted precipitation for the best sample
pred_precip_norm = pred_all_norm[best_idx]
pred_precip_actual = minmaxdenorm([pred_precip_norm], outputs_lb, outputs_ub)[1]

println("Best parameters (actual scale):")
param_names = ["orography_scale", "mountain_height", "mountain_size", "mountain_lon", "mountain_lat", 
               "temperature_equator", "temperature_pole", "temperature_usa", "temperature_pa", "zonal_wind"]
for (name, val) in zip(param_names, best_params_actual)
    println("  $name: $val")
end

println("Predicted max precipitation: $pred_precip_actual")
println("Actual max precipitation: $actual_precip")
println("Prediction error: $(abs(pred_precip_actual - actual_precip))")

println("\nAll 5 optimized samples:")
for i in 1:5
    params_i = minmaxdenorm(reshape(X_optimal[:, i], :, 1), inputs_lb, inputs_ub)[:, 1]
    pred_i = minmaxdenorm([pred_all_norm[i]], outputs_lb, outputs_ub)[1]
    println("Sample $i: predicted precipitation = $pred_i")
end
