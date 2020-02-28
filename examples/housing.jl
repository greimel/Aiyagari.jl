# # A model with homeowners and renters

using Test #src
using Aiyagari
using QuantEcon, Parameters, Interpolations
using Plots #src

# Define utility function for the case of two goods (consumption good and housing)

u(c; γ) = c^(1-γ) / (1-γ)

function u(c,h; ξ=0.8159, ρ=map(s -> (s-1)/s, 0.13), γ=2.0)
  C = (ξ * h^ρ + (1-ξ) * c^ρ)^(1/ρ)
  u(C, γ=γ)
end

# ## State space

# Exogenous states (incomes)

z_grid = [0.5; 1.0; 1.5]
z_prob = [0.7 0.15 0.15;
          0.2 0.6 0.2;
          0.15 0.15 0.7]
z_MC = MarkovChain(z_prob, z_grid, :z)

exo = ExogenousStateSpace([z_MC])

a_grid_own = LinRange(0.0, 3.0, 40)
endo = EndogenousStateSpace((w=a_grid_own,))

# ## Aggregate state

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

# ## Read relevant files

include("../housing-simple-nlopt.jl")
include("../renting-nlopt.jl")

# # Case 1: Forever home owners

r_own = 0.15
p_own = 0.9
param_own  = (β = 0.9, θ = 0.9, δ = 0.1, h_thres = eps())
agg_state_own = HousingAS(r_own, p_own, a_grid_own, exo, param_own)

@unpack policies_full, val = solve_bellman(endo, exo, agg_state_own, param_own, Owner(Aiyagari.Unconditional()), tol=1e-7)

using DelimitedFiles #src

@testset "regression test simple housing" begin #src
  #writedlm("test/matrices/housing_simple_nlopt_value.txt", val) #src
  value_test = readdlm("matrices/housing_simple_nlopt_value.txt") #src
  @test maximum(abs, val .- value_test) .< 1e-6 #src
end #src

# 

plt_w = plot(a_grid_own, policies_full.w_next, title="wealth next period", xlab="wealth")
plt_m_zoomed = plot(a_grid_own[1:15], policies_full.m[1:15,:], title="mortgage (zoomed)", xlab="wealth")

plt_m = plot(a_grid_own, policies_full.m, title="mortgage", xlab="wealth")

dist = stationary_distribution(z_MC, a_grid_own, policies_full.w_next)

plt_dist = plot(a_grid_own, dist, xlab="wealth", title="stationary distribution")

plot(plt_w, plt_dist, plt_m, plt_m_zoomed, legend=false)
excess_demand(0.15, 0.9)

#writedlm("test/matrices/housing_simple_nlopt_dist.txt", dist)
dist_test = readdlm("test/matrices/housing_simple_nlopt_dist.txt")
all(dist_test .≈ dist) || maximum(abs, dist_test .- dist)



function excess_demand(r, p)
  agg_state_own = HousingAS(r, p, a_grid_own, exo, param_own)

  @unpack policies_full, val = solve_bellman(endo, exo, agg_state_own, param_own, Owner(Aiyagari.Unconditional()), tol=1e-6)

  dist = stationary_distribution(z_MC, a_grid_own, policies_full.w_next)

  (m = sum(dist .* policies_full.m), h=sum(dist .* policies_full.h))
end

excess_demand(0.70, 0.5)


# ## Forever renters
r_rent = 0.15

a_grid_rent = LinRange(-0.05, 1.5, 50)
param_rent = (β = 0.7, θ = 0.9, δ = 0.1, h_thres = Inf)
agg_state_rent = HousingAS(r_rent, 2.2, a_grid_rent, exo, param_rent, ρ=2.2 * (param_rent.δ + r))

@unpack val, policy, policies_full = solve_bellman(a_grid_rent, exo, agg_state_rent, param_rent, Renter(Aiyagari.Unconditional()), maxiter=100)

plot(a_grid_rent, Aiyagari.get_cond_𝔼V(val, exo, 1, :z => 1))
 plot!(a_grid_rent, Aiyagari.get_cond_𝔼V(val, exo, 2, :z => 2))
 plot!(a_grid_rent, Aiyagari.get_cond_𝔼V(val, exo, 3, :z => 3))

plot(a_grid_rent, val)

plot(a_grid_rent, policies_full.w_next, title="house size", xlab="wealth", legend=:topleft)

# ### Stationary distribution

dist = stationary_distribution(exo.mc, a_grid_rent, policies_full.w_next)
plot(a_grid_rent, dist, xlab="wealth" )

# ## Own big, rent small

r = 0.10
a_grid = LinRange(0.0, 1.0, 40)
param_both = (β = 0.7, θ = 0.9, δ = 0.1, h_thres = 0.75)
param = [param_own, param_rent]
agg_state_both = HousingAS(r, 2.2, a_grid, exo, param_both, ρ= 1.07 * 2.2 * (param_both.δ + r))


 @unpack val, policy, policies_full, owner = solve_bellman(a_grid, exo, agg_state_both, param, OwnOrRent(), maxiter=70)

   plot(a_grid, owner, title="Who owns?")

# using StructArrays
# move = StructArray(exo.grid).move
# move_long = repeat(permutedims(move), 40, 1)

w_next_all = policies_full[1].w_next .* owner .+ policies_full[2].w_next .* .!owner
 h_all = policies_full[1].h .* owner .+ policies_full[2].h .* .!owner
 c_all = policies_full[1].c .* owner .+ policies_full[2].c .* .!owner

 scatter(a_grid, h_all .* (owner .== 1), color=:blue, legend=:false, title="House size", alpha=0.3, markerstrokewidth=0, xlab="wealth")
 scatter!(a_grid, h_all .* (owner .== 0), color=:red, alpha=0.3, markerstrokewidth=0)
 ylims!(0.3, 2.25)
 title!("no moving costs (blue == own, red == rent)")

savefig("no-moving-costs.png")
hline!([param_both.h_thres], color=:gray, label="", linestyle=:dash)

plot(a_grid, w_next_all, legend=false, title="wealth")

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


# scatter(a_grid, policies_full[1].m .* owner)
# scatter(a_grid, c_all)
# 
# all(policies_full.conv)

# ## Stationary distribution 

dist = stationary_distribution(exo.mc, a_grid, w_next_all)
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

