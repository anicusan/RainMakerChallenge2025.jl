using JLSO
using LinearAlgebra
using Distributions
using NNlib
using Random
using Optimization
using OptimizationOptimJL
using Flux
using CUDA
using RainMakerChallenge2025
CUDA.device!(0)

minmaxnorm(data, lb, ub, norm_min=0., norm_max=1.) = @. norm_min + (data - lb) * (norm_max - norm_min) / (ub - lb)
minmaxdenorm(data, lb, ub, norm_min=0., norm_max=1.) = @. lb + (data - norm_min) * (ub - lb) / (norm_max - norm_min)

# path = joinpath(dirname(@__DIR__), "data", "10kdata.jlso")
# data = JLSO.load(path)

# input_data = data[:d][:inputs]
# output_data = data[:d][:outputs]

# inputs_lb = [0., -2000., 0., -180., -90., 270., 270., -5., -5., 5.]
# inputs_ub = [2., 5000., 30., 180., 90., 300., 300., 5., 5., 50.]
# input_data_norm = minmaxnorm(input_data, inputs_lb, inputs_ub)

# output_data = reshape(output_data, (1,:))
# outputs_lb, outputs_ub = extrema(output_data)
# output_data_norm = minmaxnorm(output_data, outputs_lb, outputs_ub)

# tsplit = 9000
# input_data_norm_train = input_data_norm[:, 1:tsplit]
# input_data_norm_valid = input_data_norm[:, tsplit+1:end]
# output_data_norm_train = output_data_norm[:, 1:tsplit]
# output_data_norm_valid = output_data_norm[:, tsplit+1:end]

# bs = 512*2
# dataloader = Flux.DataLoader(
#     (input_data = input_data_norm_train |> Flux.gpu,
#      output_data = output_data_norm_train |> Flux.gpu);
#     batchsize = bs
# )

# act = gelu
# HSIZE = 128
# INSIZE = length(inputs_lb)
# OUTSIZE = length(outputs_lb)
# function NNLayer(in_size, out_size, act = identity; bias = false)
#     d = Uniform(-1.0 / sqrt(in_size), 1.0 / sqrt(in_size))
#     Dense(in_size, out_size, init = (x,y)->rand(d,x,y), act, bias = bias)
# end
# surrogate = Chain(
#     NNLayer(INSIZE, HSIZE, act),
#     SkipConnection(Chain(NNLayer(HSIZE, HSIZE, act),
#                          NNLayer(HSIZE, HSIZE, act)), +),
#     SkipConnection(Chain(NNLayer(HSIZE, HSIZE, act),
#                          NNLayer(HSIZE, HSIZE, act)), +),
#     NNLayer(HSIZE, OUTSIZE)
# ) |> gpu

# loss_t = []
# loss_v = []
# # Training Loop	opt = OptimiserChain(Adam())
# opt = OptimiserChain(Flux.Adam())
# lrs = [1e-3, 5e-4, 3e-4, 1e-4]
# epochs = [500 for i in 1:length(lrs)]
# st = Optimisers.setup(opt, surrogate)
# ps = Flux.params(surrogate) # to use for L2 reg
# lambda_reg = 1e-6

# for (e, lr) in zip(epochs, lrs)
#     for epoch in 1:e
#         Optimisers.adjust!(st, lr)
#         for batch in dataloader
#             x, y = batch.input_data, batch.output_data
#             gs = gradient(surrogate) do model
#                 Flux.mae(model(x), y) + lambda_reg * sum(x_ -> sum(abs2, x_), ps) #l2 reg
#             end
#             st, surrogate = Optimisers.update(st, surrogate, gs...)
#         end
#         if epoch % 100 == 0
#             surrogate_cpu = surrogate |> cpu
#             l_t = Flux.mae(surrogate_cpu(input_data_norm_train), output_data_norm_train)
#             surr_v_pred = minmaxdenorm(surrogate_cpu(input_data_norm_valid), outputs_lb, outputs_ub)
#             gt_v = minmaxdenorm(output_data_norm_valid, outputs_lb, outputs_ub)
#             l_v = Flux.mae(surr_v_pred, gt_v)
#             push!(loss_t,l_t)
#             push!(loss_v,l_v)
#             @info "Epoch $epoch lr:$lr Training Loss: $l_t Validation Loss:$l_v"
#         end
#     end
# end

# surr_cpu = surrogate |> cpu
# JLSO.save("surrogate_nn_ex_128_500-v2.jlso", Dict(:surrogate => surr_cpu))

# Simple SciML Optimization equivalent
function objective_func_simple(x, p)
    X = reshape(x, 10, :) |> gpu  # Same as your Flux code
    y_pred = surrogate(X)
    obj = -sum(y_pred)  # Same objective
    λ = 100.0f0
    lower_violation = max.(0.0f0, -x)
    upper_violation = max.(0.0f0, x .- 1.0f0)
    penalty = λ * (sum(lower_violation.^2) + sum(upper_violation.^2))
    return obj + penalty
end

# Same initial setup as your Flux code
x0 = vec(rand(Float32, 10, 10))  # Flatten to vector for Optimization.jl
optf = Optimization.OptimizationFunction(objective_func_simple, Optimization.AutoZygote())
prob = Optimization.OptimizationProblem(optf, x0)

# Progress callback (simpler)
function simple_callback(state, loss_val)
    iter = state.iter
    if iter % 1 == 0
        println("Step $iter, objective = $loss_val")
    end
    return false
end

println("Starting SciML optimization (1000 samples)...")
sol = solve(prob, LBFGS(), callback=simple_callback, maxiters=20000)

# Extract results exactly like your Flux code
X_candidate_sciml = reshape(sol.u, 10, :) |> gpu
idx = argmax(surrogate(X_candidate_sciml))
best_param_sample_sciml = minmaxdenorm(X_candidate_sciml[:,idx[2]]|>cpu, inputs_lb, inputs_ub)
pred_vals_sciml = minmaxdenorm(surrogate(X_candidate_sciml)[:,idx[2]]|>cpu, outputs_lb, outputs_ub)
actual_precip_sciml = max_precipitation(best_param_sample_sciml)


#v2 surrogate_nn_ex_128_500-v2.jlso
#With the above model we got these parameters:  
# 10-element Vector{Float64}:
# 2.0163631439208984
# 5013.73028755188
# -0.017217400018125772
# -22.203487157821655
# -19.583306908607483
# 282.971847653389
# 300.2009081840515
# 2.134748101234436
# 4.705338478088379
# 34.92057800292969 
#Surrogate predicted 285.05
#Actual was 210.16


# Function to get precipitation AND simulation for plotting
function max_precipitation_with_sim(parameters::AbstractVector)
    parameter_tuple = NamedTuple{PARAMETER_KEYS}(parameters)
    return max_precipitation_with_sim(parameter_tuple)
end

function max_precipitation_with_sim(parameters::NamedTuple)
    # define resolution. Use trunc=42, 63, 85, 127, ... for higher resolution, cubically slower
    spectral_grid = SpectralGrid(trunc=31, nlayers=8)

    # Define AquaPlanet ocean, for idealised sea surface temperatures
    # but don't change land-sea mask = retain real ocean basins
    ocean = AquaPlanet(spectral_grid,
                temp_equator=parameters.temperature_equator,
                temp_poles=parameters.temperature_pole)

    land_temperature = ConstantLandTemperature(spectral_grid)
    land = LandModel(spectral_grid; temperature=land_temperature)

    initial_conditions = InitialConditions(
        vordiv = ZonalWind(u₀=parameters.zonal_wind),
        temp = JablonowskiTemperature(u₀=parameters.zonal_wind),
        pres = PressureOnOrography(),
        humid = ConstantRelativeHumidity())

    orography = EarthOrography(spectral_grid, scale=parameters.orography_scale)

    # construct model
    model = PrimitiveWetModel(spectral_grid; ocean, land, initial_conditions, orography)

    # Add rain gauge, locate in Pittsburgh PA
    rain_gauge = RainGauge(spectral_grid, lond=-80, latd=40.45)
    add!(model, rain_gauge)

    # Initialize
    simulation = initialize!(model, time=DateTime(2025, 7, 22))

    # Add additional mountain
    H = parameters.mountain_height
    λ₀, φ₀, σ = parameters.mountain_lon, parameters.mountain_lat, parameters.mountain_size  
    set!(model, orography=(λ,φ) -> H*exp(-spherical_distance((λ,φ), (λ₀,φ₀), radius=360/2π)^2/2σ^2), add=true)

    # land sea surface temperature anomalies
    # 1. USA
    set!(simulation, soil_temperature=
        (λ, φ, k) -> (30 < φ < 50) && (240 < λ < 285) ? parameters.temperature_usa : 0, add=true)

    # 2. Pennsylvania
    A = parameters.temperature_pa
    λ_az, φ_az, σ_az = -80, 40.45, 4    # location [˚], size [˚] of Azores
    set!(simulation, soil_temperature=
        (λ, φ, k) -> A*exp(-spherical_distance((λ,φ), (λ_az,φ_az), radius=360/2π)^2/2σ_az^2), add=true)

    # Run simulation for 20 days
    run!(simulation, period=Day(20))

    # skip first 5 days, as is done in the RainMaker challenge
    RainMaker.skip!(rain_gauge, Day(5))

    # evaluate rain gauge
    lsc = rain_gauge.accumulated_rain_large_scale
    conv = rain_gauge.accumulated_rain_convection
    total_precip = maximum(lsc) + maximum(conv)
    
    return total_precip, simulation
end

# Plot meteogram for the optimized parameters
function plot_meteogram_for_optimized_params(best_param_sample_sciml)
    println("\n" * "="^60)
    println("Generating meteogram for optimized parameters...")
    println("="^60)
    
    # Use the best parameters found by SciML optimization
    println("Running simulation with optimized parameters...")
    precip, simulation = max_precipitation_with_sim(best_param_sample_sciml)
    
    println("Precipitation from simulation: $precip")
    
    # Check what callbacks are available
    println("Available callbacks:")
    for (key, callback) in simulation.model.callbacks
        println("  $key: $(typeof(callback))")
    end
    
    # Find the rain gauge callback and plot it
    rain_gauge_key = nothing
    for (key, callback) in simulation.model.callbacks
        if isa(callback, RainGauge)
            rain_gauge_key = key
            break
        end
    end
    
    if rain_gauge_key !== nothing
        println("Plotting meteogram using callback: $rain_gauge_key")
        try
            # This should create the meteogram plot
            fig = RainMaker.plot(simulation.model.callbacks[rain_gauge_key])
            println("Meteogram plotted successfully!")
            return fig
        catch e
            println("Error plotting meteogram: $e")
            println("You can manually plot with: RainMaker.plot(simulation.model.callbacks[:$rain_gauge_key])")
        end
    else
        println("No RainGauge callback found!")
        println("Available callbacks: $(keys(simulation.model.callbacks))")
    end
end

# Call the plotting function
fig = plot_meteogram_for_optimized_params(best_param_sample_sciml)
display(fig)








