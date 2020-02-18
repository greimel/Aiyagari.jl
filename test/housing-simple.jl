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

mutable struct HousingAS{T1,T2,T3} <: AggregateState
  r::T1
  p::T2
  dist::T3 # the distribution over idiosynchratic states
end

function HousingAS(p, r, a_grid, z_MC)
  dist_proto = zeros((length(a_grid), length(z_MC.state_values)))
  HousingAS(p, r, dist_proto)
end

a_grid = LinRange(0.0, 1.0, 40)
agg_state = HousingAS(0.01, 2.2, a_grid, z_MC)
param = (β = 0.7, θ = 0.9, δ = 0.1)

include("housing-simple-nlopt.jl")
include("housing-simple-jump.jl")

@time @unpack val, policy, policies_full = solve_bellman(a_grid, z_MC, agg_state, param)
# 2.5 s with NLopt
# 129 s 60 itr with JuMP


using DelimitedFiles
#writedlm("test/matrices/housing_simple_value.txt", value)
value_test = readdlm("test/matrices/housing_simple_value.txt")

@test all(value .== value_test)


using Plots, StructArrays

plot(val)
policies_SoA = StructArray(policies_full)

scatter(a_grid, policies_SoA.w_next)
scatter(a_grid, policies_SoA.h_next)
scatter(a_grid, policies_SoA.m_next)
scatter(a_grid, policies_SoA.c)

policies_SoA.ret
all(policies_SoA.conv)

dist = stationary_distribution(z_MC, a_grid, policies_SoA.w_next)

using StatsBase
mean(vec(policies_SoA.m_next), Weights(vec(dist)))
mean(vec(policies_SoA.h_next), Weights(vec(dist)))
#926 μs
plot(a_grid, dist)

#writedlm("test/matrices/huggett_dist.txt", dist)
dist_test = readdlm("test/matrices/huggett_dist.txt")
@test all(dist .== dist_test)
