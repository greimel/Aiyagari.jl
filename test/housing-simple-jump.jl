using Revise
using Test, Aiyagari
using JuMP, Ipopt, NLopt
using QuantEcon, Parameters, Interpolations

# Exogenous states (incomes)
z_grid = [0.5; 1.0; 1.5]
z_prob = [0.7 0.15 0.15;
          0.2 0.6 0.2;
          0.15 0.15 0.7]
z_MC = MarkovChain(z_prob, z_grid)

u(c; γ=γ) = c^(1-γ) / (1-γ)

function u(c,h; ξ=0.8159, ρ=map(s -> (s-1)/s, 0.13), γ=2.0)
  C = (ξ * h^ρ + (1-ξ) * c^ρ)^(1/ρ)
  u(C, γ=γ)
end

function Aiyagari.get_optimum(states, agg_state, 𝔼V, params, a_grid)
  @unpack p, r = agg_state
  @unpack β, θ, δ = params
  
  model = Model()
  set_optimizer(model, NLopt.OpNLoptSolver(algorithm=:LD_MMA))
  
  MOI.set(model, MOI.RawParameter("print_level"), 0)
  #model = Model(with_optimizer(Ipopt.Optimizer, print_level=0))

  register(model, :u, 2, u, autodiff=true)
  register(model, :𝔼V, 1, w -> 𝔼V(w), autodiff=true)

  @variable(model, c >= eps())
  @variable(model, h >= eps())  

  @NLparameter(model, w == states.a)
  @NLparameter(model, y == states.z)
  
  w_next = @NLexpression(model, w + y - c - p * h * (r + δ))
  m = @NLexpression(model, p * h + c - y - w)
  @NLconstraint(model, (1+r) * m <= p * h * (1-δ) * θ)
  @NLobjective(model, Max, u(c,h) + β * 𝔼V(w_next) )
  
  MOI.set(model, MOI.Silent(), true)
  
  optimize!(model)
  
  val = objective_value(model)

  pol_full = (c=value(c), h=value(h), m=value(m), w_next=value(w_next), conv=termination_status(model))
  pol = pol_full.w_next, pol_full.h                    
                  
  conv = termination_status(model) == 4
      
  (pol=pol, pol_full=pol_full, val=val, conv=conv)
  

end

# constrained optimization is provided in Optim!
#https://julianlsolvers.github.io/Optim.jl/stable/#examples/generated/ipnewton_basics/#nonlinear-constrained-optimization
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

#using BenchmarkTool
@time @unpack val, policy, policies_full = solve_bellman(a_grid, z_MC, agg_state, param)
# 3.9 s 56 itr (n=40)
# 5.7 s 56 itr (n=40)
# 129 s 60 itr with JuMP

using DelimitedFiles
#writedlm("test/matrices/housing_simple_value.txt", value)
value_test = readdlm("test/matrices/housing_simple_value.txt")

@test all(value .== value_test)


using Plots, StructArrays

plot(val)
policies_SoA = StructArray(policies_full)

scatter(a_grid, policies_SoA.w_next)
scatter(a_grid, policies_SoA.h)
scatter(a_grid, policies_SoA.m)
scatter(a_grid, policies_SoA.c)


dist = stationary_distribution(z_MC, a_grid, policies_SoA.w_next)

using StatsBase
mean(vec(policies_SoA.m), Weights(vec(dist)))
mean(vec(policies_SoA.h), Weights(vec(dist)))
#926 μs
plot(a_grid, dist)

#writedlm("test/matrices/huggett_dist.txt", dist)
dist_test = readdlm("test/matrices/huggett_dist.txt")
@test all(dist .== dist_test)
