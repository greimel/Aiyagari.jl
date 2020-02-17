using Revise
using Test, Aiyagari
using Optim, QuantEcon, Parameters, Interpolations

# Exogenous states (incomes)
z_grid = [0.5; 1.0; 1.5]
z_prob = [0.7 0.15 0.15;
          0.2 0.6 0.2;
          0.15 0.15 0.7]
z_MC = MarkovChain(z_prob, z_grid)


u(c; γ=γ) = c > 0 ? c^(1-γ) / (1-γ) : 10000 * c - 10000 * one(c)

@inline function u(c,h; ξ=0.8159, ρ=map(s -> (s-1)/s, 0.13), γ=2.0) # Housing share and Intra-temporal ES (0.13) from Garriga & Hedlund
  C = (ξ * h^ρ + (1-ξ) * c^ρ)^(1/ρ)
  u(C, γ=γ)
end

function transform_policies(plain_action, plain_state, prices, tech)
    @unpack h_next, w_next = plain_action
    @unpack y, w = plain_state
    @unpack p, r = prices
    @unpack θ, δ = tech
    κ₀ = (w+y)*(1+r) / (p * (1+r-θ))
    α₁ = h_next / κ₀
    κ₁ = (w+y) - p * h_next * δ
    κ₂ = p * h_next * (1-δ - θ)
    α₂ = (w_next - κ₂) / (κ₁ - κ₂)
    (α₁=α₁, α₂=α₂)
end

function get_back_policies(transf_action, plain_state, prices, tech)
    @unpack α₁, α₂ = transf_action
    @unpack y, w = plain_state
    @unpack p, r = prices
    @unpack θ, δ = tech
    κ₀ = (w+y)*(1+r) / (p * (1+r-θ))
    h_next = α₁ * κ₀

    κ₁ = (w+y) - p * h_next * δ
    κ₂ = p * h_next * (1-δ - θ)

    w_next = α₂ *(κ₁ - κ₂) + κ₂
    (w_next=w_next, h_next=h_next)
end

function all_policies(plain_action, plain_state, prices, tech)
  @unpack h_next, w_next = plain_action
  @unpack y, w = plain_state
  @unpack p, r = prices
    @unpack θ, δ, β = tech
    
  ϵ = eps(typeof(h_next))

  h_next = h_next < ϵ ? ϵ : h_next
  m_next = (1-δ) * p * h_next - w_next
  c = w + y + m_next / (1+r) - p * h_next
  
  (h_next=h_next, w_next=w_next, m_next=m_next, c=c)
  #(h_next=h_next, c=c)
end

function obj(plain_action, plain_state, prices, tech, value_itp)
    @unpack w_next = plain_action
    @unpack β = tech

    @unpack h_next, c = all_policies(plain_action, plain_state, prices, tech)
    
    c > 0 ? u(c, h_next) + β * value_itp(w_next) : - 10_000 + 10_000 * c
    #u(c, h_next) + β * value_itp(w_next)
end

function obj_bounded(α₁, α₂, plain_state, prices, tech, value_itp)
    plain_action = get_back_policies((α₁=α₁, α₂=α₂), plain_state, prices, tech)
    obj(plain_action, plain_state, prices, tech, value_itp)
end

function Aiyagari.get_optimum(states, agg_state, 𝔼V, params, a_grid)
  plain_state = (w = states.a, y = states.z)
  lower = [eps(); eps()]
  upper = [1-eps(); 1-eps()]
  initial_x = [0.5; 0.5]
  
  res = optimize(α -> -obj_bounded(α[1], α[2], plain_state,
                     agg_state, params, 𝔼V),
                 lower, upper, initial_x, Fminbox(), autodiff=:forward)
                 
  conv = any([res.f_converged; res.x_converged; res.g_converged])
  val  = - Optim.minimum(res)
  α₁, α₂ = res.minimizer
  transf_action = (α₁=α₁, α₂=α₂)
  plain_action =  get_back_policies(transf_action, plain_state, agg_state, params)
  
  other_actions = all_policies(plain_action, plain_state, agg_state, params)
      
  (pol=collect(plain_action), pol_full=other_actions, val=val, conv=conv)
  

end

mutable struct HousingAS{T1,T2,T3} <: AggregateState
  r::T1
  p::T2
  dist::T3 # the distribution over idiosynchratic states
end

function HousingAS(p, r, a_grid, z_MC)
  dist_proto = zeros((length(a_grid), length(z_MC.state_values)))
  HousingAS(p, r, dist_proto)
end

a_grid = LinRange(0.01, 10, 40)
agg_state = HousingAS(2.2, 0.01, a_grid, z_MC)
param = (β = 0.7, θ = 0.9, δ = 0.2)

#using BenchmarkTool
@time @unpack value, policy, policies_full = solve_bellman(a_grid, z_MC, agg_state, param)
# 3.9 s 56 itr (n=40)
# 5.7 s 56 itr (n=40)

using DelimitedFiles
value_test = readdlm("test/matrices/housing_simple_value.txt")
#writedlm("test/matrices/housing_simple_value.txt", value)

@test all(value .== value_test)


using Plots, StructArrays

plot(value)
policies_SoA = StructArray(policies_full)

plot(a_grid, policies_SoA.w_next)
plot(a_grid, policies_SoA.h_next)

dist = stationary_distribution(z_MC, a_grid, policies_SoA.w_next)
#926 μs

#writedlm("test/matrices/huggett_dist.txt", dist)
dist_test = readdlm("test/matrices/huggett_dist.txt")
@test all(dist .== dist_test)
