using Revise
using Test, Aiyagari
using QuantEcon, Parameters, Interpolations


u(c; γ=γ) = c^(1-γ) / (1-γ)

function u(c,h; ξ=0.8159, ρ=map(s -> (s-1)/s, 0.13), γ=2.0)
  C = (ξ * h^ρ + (1-ξ) * c^ρ)^(1/ρ)
  u(C, γ=γ)
end

# Exogenous states (incomes)
z_grid = [0.5; 1.0; 1.5]
z_prob = [0.7 0.15 0.15;
          0.2 0.6 0.2;
          0.15 0.15 0.7]
z_MC = MarkovChain(z_prob, z_grid)

mutable struct HousingAS{T1,T2,T3,T4} <: AggregateState
  r::T1
  p::T2 # house price
  ρ::T3 # rent
  dist::T4 # the distribution over idiosynchratic states
end

function HousingAS(r, p, a_grid, z_MC, param)
  dist_proto = zeros((length(a_grid), length(z_MC.state_values)))
  ρ = p * (param.δ)
  HousingAS(r, p, ρ, dist_proto)
end

#include("housing-simple-nlopt.jl")
include("renting-nlopt.jl")
#include("housing-simple-jump.jl")
r = 0.29
 a_grid = LinRange(-√eps(), 5, 50)
 param = (β = 0.7, θ = 0.9, δ = 0.1)
 agg_state = HousingAS(r, 2.2, a_grid, z_MC, param)

@unpack val, policy, policies_full = solve_bellman(a_grid, z_MC, agg_state, param)
 # 2.5 s with NLopt
 # 129 s 60 itr with JuMP
scatter(a_grid, policies_full.a_next)
  plot!(a_grid, a_grid)
  plot!(a_grid[[1;end]], a_grid[[end;end]], legend=false)

using Plots
using DelimitedFiles
#writedlm("test/matrices/housing_simple_nlopt_value.txt", val)
value_test = readdlm("test/matrices/housing_simple_nlopt_value.txt")

@test all(val .== value_test)

using Plots
plot(val)

scatter(a_grid, policies_full.w_next)
scatter(a_grid, policies_full.h)
scatter(a_grid, policies_full.m)
scatter(a_grid, policies_full.c)

all(policies_full.conv)

dist = stationary_distribution(z_MC, a_grid, policies_full.a_next)
 plot(a_grid, dist)
#writedlm("test/matrices/housing_simple_nlopt_dist.txt", dist)
dist_test = readdlm("test/matrices/housing_simple_nlopt_dist.txt")
@test all(dist .== dist_test)

using StatsBase
mean(vec(policies_full.m), Weights(vec(dist)))
mean(vec(policies_full.h), Weights(vec(dist)))
#926 μs


