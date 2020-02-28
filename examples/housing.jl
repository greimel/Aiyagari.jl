# # A model with homeowners and renters

#jl using Test
using Aiyagari
using QuantEcon, Parameters
#md using Plots

PKG_HOME = joinpath(dirname(pathof(Aiyagari)), "..")

# ## Setting up the state space

# Exogenous states (incomes)

z_grid = [0.5; 1.0; 1.5]
z_prob = [0.7 0.15 0.15;
          0.2 0.6 0.2;
          0.15 0.15 0.7]
z_MC = MarkovChain(z_prob, z_grid, :z)

exo = ExogenousStateSpace([z_MC])

# ## Define utility function for the case of two goods (consumption good and housing)

u(c; γ) = c^(1-γ) / (1-γ)

function u(c,h; ξ=0.8159, ρ=map(s -> (s-1)/s, 0.13), γ=2.0)
  C = (ξ * h^ρ + (1-ξ) * c^ρ)^(1/ρ)
  u(C, γ=γ)
end

include(joinpath(PKG_HOME, "test", "housing-simple-nlopt.jl"))
include(joinpath(PKG_HOME, "test", "renting-nlopt.jl"))

# ## Aggregate state

mutable struct HousingAS{T1,T2,T3,T4} <: AggregateState
  r::T1
  p::T2 # house price
  ρ::T3 # rent
  dist::T4 # the distribution over idiosynchratic states
end

function HousingAS(r, p, w_grid, exo, param; ρ=p * (param.δ + r))
  dist_proto = zeros((length(w_grid), length(exo))) 
  HousingAS(r, p, ρ, dist_proto)
end

# # Case 1: Forever home owners

w_grid_own = LinRange(0.0, 3.0, 40)
endo_own = EndogenousStateSpace((w=w_grid_own,))

r_own = 0.15
p_own = 0.9
param_own  = (β = 0.9, θ = 0.9, δ = 0.1, h_thres = eps())
agg_state_own = HousingAS(r_own, p_own, w_grid_own, exo, param_own)

@unpack policies_full, val = solve_bellman(endo_own, exo, agg_state_own, param_own, Owner(Aiyagari.Unconditional()), tol=1e-7)

dist = stationary_distribution(exo.mc, endo_own.grids.w, policies_full.w_next)

#jl using DelimitedFiles

# 

#md plt_w = plot(w_grid_own, policies_full.w_next, title="wealth next period", xlab="wealth")
#md plt_m_zoomed = plot(w_grid_own[1:15], policies_full.m[1:15,:], title="mortgage (zoomed)", xlab="wealth")
 
#md plt_m = plot(w_grid_own, policies_full.m, title="mortgage", xlab="wealth")


#md plt_dist = plot(w_grid_own, dist, xlab="wealth", title="stationary distribution")

#md plot(plt_w, plt_dist, plt_m, plt_m_zoomed, legend=false)

#jl # using DelimitedFiles
#jl # writedlm(joinpath(PKG_HOME, "test/matrices", "housing-simple-value.txt"), val) #src
#jl # writedlm(joinpath(PKG_HOME, "test/matrices", "housing-simple-dist.txt"), dist) #src

#jl @testset "regression test housing-simple" begin
#jl   value_test = readdlm(joinpath(PKG_HOME, "test/matrices", "housing-simple-value.txt"))
#jl   dist_test = readdlm(joinpath(PKG_HOME, "test/matrices", "housing-simple-dist.txt"))
#jl   @test all(val .== value_test)
#jl   @test maximum(abs, dist .- dist_test) < 1e-12
#jl end

# ## Equilibrium

# First, let's compute excess demand

function excess_demand(r, p)
  agg_state_own = HousingAS(r, p, w_grid_own, exo, param_own)

  @unpack policies_full, val = solve_bellman(endo_own, exo, agg_state_own, param_own, Owner(Aiyagari.Unconditional()), tol=1e-6)

  dist = stationary_distribution(z_MC, w_grid_own, policies_full.w_next)

  (m = sum(dist .* policies_full.m), h=sum(dist .* policies_full.h))
end

excess_demand(r_own, p_own)

# TODO: Housing supply

# # Own big, rent small

# Now we want to give households to choice whether to buy or to rent

function combined_policies(policies_full, owner, own, rent=own; f_own =identity, f_rent=identity)
  f_own.(getproperty(policies_full[1], own)) .* owner .+ f_rent.(getproperty(policies_full[2], rent)) .* .!(owner) 
end

function excess_demand(r, p, endo, exo, param, hh; maxiter_bellman=200)
  agg_state = HousingAS(r, p, endo.grids.w, exo, param, ρ= p * (param[1].δ + r))

  out = solve_bellman(endo, exo, agg_state, param, hh, maxiter=300, tol=1e-6)
  
  @unpack val, policy, policies_full, owner = out
  
  w_next_all = combined_policies(policies_full, owner, :w_next)
  
  dist = stationary_distribution(exo.mc, w_grid, w_next_all)
  a_all = combined_policies(policies_full, owner, :m, :w_next, f_own = x -> -x)
  h_all = combined_policies(policies_full, owner, :h)
  
  (a = sum(dist .* a_all), h=sum(dist .* h_all), dist=dist, out...)
end

r = 0.01
p = 1.7
w_grid = LinRange(0.0, 4.0, 40)
endo = EndogenousStateSpace((w=w_grid,))

param_both = (β = 0.93, θ = 0.9, δ = 0.1, h_thres = 1.2)
param = [param_both, param_both]

@unpack val, policy, policies_full, owner, a, h, dist = excess_demand(r, p, endo, exo, param, OwnOrRent(Owner(Aiyagari.Unconditional()), Renter()))

@show r => a, p => h
 
#md plt_own = plot(w_grid, owner, title="Who owns?")

w_next_all = combined_policies(policies_full, owner, :w_next)
h_all = combined_policies(policies_full, owner, :h)
c_all = combined_policies(policies_full, owner, :c)
a_all = combined_policies(policies_full, owner, :m, :w_next, f_own = x -> -x)

#md plt_h = plot(w_grid, h_all, legend=:false, title="House size", markerstrokewidth=0, xlab="wealth")

#md plt_a = plot(w_grid, a_all, title = "asset/mortgage")

#md plt_w = plot(w_grid, w_next_all, title = "wealth next period")

#md plt_dist = plot(w_grid, dist, title = "stationary distribution")

#md plot(plt_h, plt_a, plt_w, plt_dist, legend=false)

#

#writedlm(joinpath(PKG_HOME, "test/matrices", "own-rent-value.txt"), val) #src
#writedlm(joinpath(PKG_HOME, "test/matrices", "own-rent-dist.txt"), dist) #src

#jl @testset "own-rent" begin
#jl   value_test = readdlm(joinpath(PKG_HOME, "test/matrices", "own-rent-value.txt"))
#jl   dist_test = readdlm(joinpath(PKG_HOME, "test/matrices", "own-rent-dist.txt"))
#jl   @test all(val .≈ value_test)
#jl   @test maximum(abs, dist .- dist_test) < 1e-12
#jl end


