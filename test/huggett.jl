using Test, Aiyagari
using Optim, QuantEcon, Parameters, Interpolations

a_grid = LinRange(-2, 5, 20)

# Exogenous states (incomes)
z_grid = [0.5; 1.0; 1.5]
z_prob = [0.7 0.15 0.15;
          0.2 0.6 0.2;
          0.15 0.15 0.7]
z_MC = MarkovChain(z_prob, z_grid)

u(c) = c > 0 ? log(c) : 10000 * c - 10000 * one(c)

function Aiyagari.iterate_bellman!(value_new, value_old, policy, policies_full, a_grid, z_mc, converged, agg_state)
  @unpack r = agg_state
  β = 0.9
  a_min, a_max = extrema(a_grid)
  
  for (i_z, z) in enumerate(z_mc.state_values)
    # Create interpolated expected value function
    exp_value = value_old * z_mc.p[i_z,:]

    itp_exp_value = interpolate(exp_value, BSpline(Cubic(Line(OnGrid()))))
    sitp_exp_value = scale(itp_exp_value, a_grid)
    
    for (i_a, a) in enumerate(a_grid) 
      
      obj = a_next -> begin
        c = a + z - a_next/(1+r)
        u(c) + β * sitp_exp_value(a_next)
      end

      res = optimize(a_next -> - obj(a_next), a_min, a_max)

      policy[i_a, i_z]    = Optim.minimizer(res)
      value_new[i_a, i_z] = - Optim.minimum(res)
      converged[i_a, i_z] = Optim.converged(res)
    end
  end
end

include("../src/aggregate-state.jl")

mutable struct HuggettAS{T1,T2} <: AggregateState
  r::T1
  dist::T2 # the distribution over idiosynchratic states
end

function HuggettAS(r, a_grid, z_MC)
  dist_proto = zeros((length(a_grid), length(z_MC.state_values)))
  HuggettAS(r, dist_proto)
end

agg_state = HuggettAS(0.05, a_grid, z_MC)

#using BenchmarkTools
@unpack value, policy = solve_bellman(a_grid, z_MC, (), agg_state)
# 22 ms
using DelimitedFiles
#writedlm("test/matrices/huggett_value.txt", value)
value_test = readdlm("test/matrices/huggett_value.txt")

@test all(value .== value_test)

a_min, a_max = a_grid[[1;end]]
@assert all(a_min .< policy .< a_max)

using Plots
plot(a_grid, value)
plot(a_grid, policy)

using BenchmarkTools

let value = value, policy=policy, z_mc = z_MC
  lin_ind = LinearIndices(size(value))

  ngp_exo = length(z_mc.state_values)
  n = length(policy)
  len = n * ngp_exo * 2

  I = zeros(Int, len)
  J = zeros(Int, len)
  V = zeros(len)

  @btime Aiyagari.controlled_markov_chain!($I, $J, $V, $lin_ind, $z_mc, $a_grid, $policy)
end

dist = stationary_distribution(z_MC, a_grid, policy)
#926 μs

#writedlm("test/matrices/huggett_dist.txt", dist)
dist_test = readdlm("test/matrices/huggett_dist.txt")
@test all(dist .== dist_test)

using Plots
plot(a_grid, reshape(dist, size(value)))

