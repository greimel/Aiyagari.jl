# # A model with homeowners and renters

using Revise
using Test, Aiyagari
using QuantEcon, Parameters, Interpolations
using Plots

u(c; γ) = c^(1-γ) / (1-γ)

function u(c,h; ξ=0.8159, ρ=map(s -> (s-1)/s, 0.13), γ=2.0)
  C = (ξ * h^ρ + (1-ξ) * c^ρ)^(1/ρ)
  u(C, γ=γ)
end

# Exogenous states (incomes)
z_grid = [0.5; 1.0; 1.5]
z_prob = [0.7 0.15 0.15;
          0.2 0.6 0.2;
          0.15 0.15 0.7]
z_MC = MarkovChain(z_prob, z_grid, :z)

# Moving shocks
move_grid = Symbol[:just_moved, :normal, :move]
move_grid = [1, 2, 3]
move_prob = [0.7 0.3 0.0;
          0.0 0.9 0.1;
          1.0 0.0 0.0]
move_MC = MarkovChain(move_prob, move_grid, :move)

# exo = ExogenousStatespace([z_MC, move_MC])

# itp_scheme = BSpline(Cubic(Line(OnGrid())))
# a_grid = LinRange(0.0, 0.7, 50)
# 
# val = u.(a_grid .+ permutedims([exo.z for exo in exo1.grid]), γ=2.0)
# 
# 𝔼V = extrapolated_𝔼V(a_grid, itp_scheme, val, exo1, 3, Aiyagari.Conditional(:move))
# 
# 𝔼V([1.0, 2.0, 0.7])

mutable struct HousingAS{T1,T2,T3,T4} <: AggregateState
  r::T1
  p::T2 # house price
  ρ::T3 # rent
  dist::T4 # the distribution over idiosynchratic states
end

function HousingAS(r, p, a_grid, exo, param; ρ=p * (param.δ + r))
  dist_proto = zeros((length(a_grid), length(exo))) 
  HousingAS(r, p, ρ, dist_proto)
end

include("housing-simple-nlopt.jl")
#include("renting-nlopt.jl")
#include("housing-simple-jump.jl")
r = 0.29

# exo_old = ExogenousStatespace([z_MC])
exo = ExogenousStatespace([z_MC, move_MC])

# ## Forever home owners
r_own = 0.29
a_grid_own = LinRange(0.0, 0.7, 50)
param_own  = (β = 0.7, θ = 0.9, δ = 0.1, h_thres = eps())
agg_state_own = HousingAS(r_own, 2.2, a_grid_own, exo, param_own)
  
@unpack val, policy, policies_full = solve_bellman(a_grid_own, exo, agg_state_own, param_own, Owner(Aiyagari.Conditional(:move)))

using DelimitedFiles

@testset "regression test simple housing" begin
  #writedlm("test/matrices/housing_simple_nlopt_value.txt", val)
  value_test = readdlm("test/matrices/housing_simple_nlopt_value.txt")

  for i in 1:length(z_grid)
    Δ = maximum(abs, value_test[:,i] .- Aiyagari.get_cond_𝔼V(val, exo, i, :z => i))
    @test abs(Δ) < 1e-6
  end
end

#@show all(value_test .≈ val) || maximum(abs, value_test .- val)

plot(a_grid_own, policies_full.h, title="house size", xlab="wealth", legend=:topleft)

# 

#plot(a_grid_own, policies_full.m, title="mortgage", xlab="wealth")

dist = stationary_distribution(z_MC, a_grid_own, policies_full.w_next)

#writedlm("test/matrices/housing_simple_nlopt_dist.txt", dist)
dist_test = readdlm("test/matrices/housing_simple_nlopt_dist.txt")
all(dist_test .≈ dist) || maximum(abs, dist_test .- dist)

plot(a_grid_own, dist, xlab="wealth" )

# ## Forever renters

a_grid_rent = LinRange(-√eps(), 5.0, 50)
param_rent = (β = 0.7, θ = 0.9, δ = 0.1, h_thres = Inf)
agg_state_rent = HousingAS(r, 2.2, a_grid_rent, z_MC, param_rent)

@unpack val, policy, policies_full = solve_bellman(a_grid_rent, z_MC, agg_state_rent, param_rent, Renter())

plot(a_grid_rent, policies_full.w_next, title="house size", xlab="wealth", legend=:topleft)

# ### Stationary distribution

dist = stationary_distribution(z_MC, a_grid_rent, policies_full.w_next)
plot(a_grid_rent, dist, xlab="wealth" )

# ## Own big, rent small

a_grid = LinRange(-√eps(), 1.5, 50)
param_both = (β = 0.7, θ = 0.9, δ = 0.1, h_thres = 0.6)
agg_state_both = HousingAS(r, 2.2, a_grid, z_MC, param_both, ρ=2.2 * (param_both.δ + r) * 1.5)


@unpack val, policy, policies_full, owner = solve_bellman(a_grid, z_MC, agg_state_both, param_both, OwnOrRent(), maxiter=70)

w_next_all = policies_full[1].w_next .* owner .+ policies_full[2].w_next .* .!owner
h_all = policies_full[1].h .* owner .+ policies_full[2].h .* .!owner
c_all = policies_full[1].c .* owner .+ policies_full[2].c .* .!owner

plot(a_grid, owner, title="Who owns?")

plot(a_grid, h_all, legend=:topleft, title="House size")

# 

# 
# plot(a_grid, w_next_all, legend=:left, title="cash-at-hand")
#   #plot!(a_grid, a_grid)
#   #plot!(a_grid[[1;end]], a_grid[[end;end]], legend=false)
# 
# using Plots
# using DelimitedFiles
# #writedlm("test/matrices/housing_simple_nlopt_value.txt", val)
# value_test = readdlm("test/matrices/housing_simple_nlopt_value.txt")
# 
# @test all(val .== value_test)
# 
# using Plots
# plot(val)
# 
# scatter(a_grid, policies_full.w_next)
# scatter(a_grid, policies_full[1].h)
# scatter!(a_grid, policies_full[2].h)

plot(a_grid, h_all, legend=:topleft, title="House size")
hline!([param_both.h_thres], color=:gray, label="", linestyle=:dash)

# scatter(a_grid, policies_full[1].m .* owner)
# scatter(a_grid, c_all)
# 
# all(policies_full.conv)

# ## Stationary distribution 

dist = stationary_distribution(z_MC, a_grid, w_next_all)
plot(a_grid, dist, xlab="wealth" )

# #writedlm("test/matrices/housing_simple_nlopt_dist.txt", dist)
# dist_test = readdlm("test/matrices/housing_simple_nlopt_dist.txt")
# @test all(dist .== dist_test)
# 
# using StatsBase
# mean(vec(policies_full.m), Weights(vec(dist)))
# mean(vec(policies_full.h), Weights(vec(dist)))
# #926 μs
# 

