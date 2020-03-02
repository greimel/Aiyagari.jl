# # Solving the Huggett model

using Test #jl
using Aiyagari
using Optim, QuantEcon, Parameters

PKG_HOME = joinpath(dirname(pathof(Aiyagari)), "..")

# ## Setting up the state space

a_grid = LinRange(-2, 5, 20)

endo = EndogenousStateSpace((a=a_grid,))

z_grid = [0.5; 1.0; 1.5]
z_prob = [0.7 0.15 0.15;
          0.2 0.6 0.2;
          0.15 0.15 0.7]
z_MC = MarkovChain(z_prob, z_grid, :z)

exo = ExogenousStateSpace([z_MC]);

# ## Define the household's problem in the Huggett model

u(c) = c > 0 ? log(c) : 10000 * c - 10000 * one(c)

function get_c(choices, states, aggregate_state)
  @unpack a_next = choices
  @unpack a, z = states
  @unpack r = aggregate_state
  
  c = a + z - a_next/(1+r)
end

function objective(choices, states, aggregate_state, 𝔼V, params)
  @unpack a_next = choices
  @unpack β = params
  
  c = get_c(choices, states, aggregate_state)

  u(c) + β * 𝔼V(a_next)
end

function Aiyagari.get_optimum(states, agg_state, 𝔼V, params, endo, hh::Consumer)
  a_grid = endo.grids.a
  
  a_min, a_max = extrema(a_grid)

  res = Optim.optimize(a_next -> - objective((a_next=a_next,), states, agg_state, 𝔼V, params), a_min, a_max)
  
  a_next = Optim.minimizer(res)
  val    = - Optim.minimum(res)
  conv   = Optim.converged(res)
  
  pol = a_next
  pol_full = (a_next=a_next, c=get_c((a_next=a_next, ), states, agg_state))
  
  (pol=pol, pol_full=pol_full, val=val, conv=conv)
end

mutable struct HuggettAS{T1,T2} <: AggregateState
  r::T1
  dist::T2 # the distribution over idiosynchratic states
end

function HuggettAS(r, a_grid, z_MC)
  dist_proto = zeros((length(a_grid), length(z_MC.state_values)))
  HuggettAS(r, dist_proto)
end;

# ## Initialize the aggregate state and solve the model

agg_state = HuggettAS(0.05, a_grid, z_MC)
param = (β = 0.9, )
  
#using BenchmarkTools
@unpack val, policy, policies_full = solve_bellman(endo, exo, agg_state, param, Consumer(), rtol=√eps())
#22 ms 176 itr

using DelimitedFiles #jl
using DelimitedFiles #src
#writedlm(joinpath(PKG_HOME, "test/matrices", "huggett_value.txt"), val) #src
value_test = readdlm(joinpath(PKG_HOME, "test/matrices", "huggett_value.txt")) #jl

@test all(val .≈ value_test) #jl

a_min, a_max = a_grid[[1;end]]
@test all(a_min .< policy .< a_max) #jl

using Plots #md
plot(a_grid, val) #md
#- #md
plot(a_grid, policies_full.a_next) #md
#- #md
plot(a_grid, policies_full.c) #md

#using BenchmarkTools #src

# let value = value, policy=policy, z_mc = z_MC #src
#   lin_ind = LinearIndices(size(value)) #src
#  #src
#   ngp_exo = length(z_mc.state_values) #src
#   n = length(policy) #src
#   len = n * ngp_exo * 2 #src
#  #src
#   I = zeros(Int, len) #src
#   J = zeros(Int, len) #src
#   V = zeros(len) #src
#  #src
#   @btime Aiyagari.controlled_markov_chain!($I, $J, $V, $lin_ind, $z_mc, #src $a_grid, $policy)
# end #src

dist = stationary_distribution(endo, exo, policy)
#926 μs

#writedlm("test/matrices/huggett_dist.txt", dist) #src
dist_test = readdlm(joinpath(PKG_HOME, "test/matrices", "huggett_dist.txt")) #jl
@test maximum(abs, dist .- dist_test) < 1e-14 #jl

# ## Looking at the equilibrium

function excess_demand(r)
  agg_state = HuggettAS(r, a_grid, z_MC)
  @unpack val, policy, policies_full = solve_bellman(endo, exo, agg_state, param, Consumer(), rtol=√eps())

  dist = stationary_distribution(endo, exo, policy)

  sum(dist .* policies_full.a_next)
end

r_vec = 0.07:0.005:0.11 #md

plot(r_vec, excess_demand.(r_vec)) #md

# ## Finite Horizon: Solving the OLG version of the model

function Aiyagari.get_optimum_last(states, agg_state, params, endo, hh::Consumer)
  a_grid = endo.grids.a
  
  a_next = 0.0
  val    = objective((a_next=a_next,), states, agg_state, x -> 0, params)
  conv   = true
  
  pol = a_next
  pol_full = (a_next=a_next, c=get_c((a_next=a_next, ), states, agg_state))
  
  (pol=pol, pol_full=pol_full, val=val, conv=conv)
end

age_grid = 25:65

@unpack val, policy, policies_full = solve_bellman(endo, exo, agg_state, param, Consumer(), age_grid)

using Plots, Colors #md
#md
plt = plot(legend=false) #md
 ran = 1:length(age_grid)- 9 #md
 col = range(colorant"deepskyblue", stop=colorant"navyblue", length=length(ran)) #md
 map(ran) do i #md
   plot!(plt, a_grid, val[:,:,i], color=col[i]) #md
 end #md
 plt #md

