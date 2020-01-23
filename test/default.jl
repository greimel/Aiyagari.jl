## Agents consume, agents save, agents get expenditure shocks
## If agents default they are excluded from financial markets

# z ... income shock 1
# η ... income shock 2
# κ ... expenditure shock

using QuantEcon, Parameters, Optim, Interpolations
include("../src/ExogenousStates.jl")
using .ExogenousStates
using Aiyagari

## Incomes z, avg income = 1
z_grid = [0.5; 1.0; 1.5]
z_prob = [0.7 0.15 0.15;
          0.15 0.7 0.15;
          0.15 0.15 0.7]
z_MC = MarkovChain(z_prob, z_grid)


## Expenditure shock
κ_grid = [0.0, 0.26, 0.82]
κ_prob = [0.925 0.07 0.005;
          0.925 0.07 0.005;
          0.925 0.07 0.005]
κ_MC = MarkovChain(κ_prob, κ_grid)

# n_exo_states = n_states(exo_MC)
# n_endo_states = length(a_grid)
# 
# V_default = zeros(n_endo_states, n_exo_states)
# V_no_default = zeros(n_endo_states, n_exo_states)

# expectation

u(c) = c > 0 ? log(c) : c * 1e5 - 1e3

q(a,z,agg_state) = (1-agg_state.θ(a,z))/(1+agg_state.r)

function Aiyagari.iterate_bellman!(value_new, value_old, policy, policies_full, a_grid, z_mc, converged, agg_state)
  β, γ = 0.95, 0.5
  a_min, a_max = extrema(a_grid)
  
  for (i_z, nt) in enumerate(z_mc.state_values)
    @unpack z, κ = nt
    # Create interpolated expected value function
    exp_value = value_old * z_mc.p[i_z,:]

    itp_exp_value = interpolate(exp_value, BSpline(Cubic(Line(OnGrid()))))
    sitp_exp_value = scale(itp_exp_value, a_grid)
    
    for (i_a, a) in enumerate(a_grid) 
      
      obj = a_next -> begin
        c = a + z - κ - q(a_next,z,agg_state) * a_next
        u(c) + β * sitp_exp_value(a_next)
      end

      res = optimize(a_next -> - obj(a_next), a_min, a_max)
      
      a_no_def = Optim.minimizer(res)
      c_no_def = a + z - κ - a_no_def * q(a_no_def,z,agg_state)
      v_no_def = - Optim.minimum(res)
      
      # default
      c_default = (1-γ) * z
      
      a_default = (a + γ * z - κ) * (1+agg_state.r)
      v_default = u(c_default) + β * extrapolate(sitp_exp_value, Interpolations.Line())(a_default)
      
      if v_no_def > v_default
        policy[i_a, i_z] = a_no_def
        policies_full[i_a, i_z]= (default = false, c = c_no_def) 
        value_new[i_a, i_z]    = v_no_def
        converged[i_a, i_z]    = Optim.converged(res)
      else
        policy[i_a, i_z] = a_default
        policies_full[i_a, i_z]= (default = true, c = c_default) 
        value_new[i_a, i_z]    = v_default
        converged[i_a, i_z]    = Optim.converged(res)
      end
    end
  end
end

using StructArrays
using StatsBase: Weights
using DataFrames

mutable struct AggStateNoFreshStart{T1, T2, T3}
  r::T1
  default_choice::T2
  θ::T3 # default_probability
end

function AggStateNoFreshStart(r, a_grid::AbstractVector, exo_mc::MarkovChain)
  size = (length(a_grid), length(exo_mc.state_values))
  
  default_choice = falses(size)
  
  AggStateNoFreshStart(r, default_choice, default_probabilities(a_grid, exo_mc))
end

function update_probabilities!(agg_state, a_grid, exo_mc, policies_full, dist)
  agg_state.default_choice .=  default_choice(policies_full)
  agg_state.θ = default_probabilities(a_grid, exo_mc, agg_state.default_choice, dist)
end

function default_choice(policies_full)
  policies_SoA = StructArray(policies_full)
  default_choice = policies_SoA.default
end

function default_probabilities(a_grid, exo_mc, default_choice, dist)
  dist = reshape(dist, size(default_choice))
    
  statespace = DataFrame([(a=a, exo...) for exo in exo_MC.state_values for a in a_grid])

  statespace[!,:pmf] = vec(dist)
  statespace[!,:default] = vec(default_choice)

  default_df = by(statespace, [:a, :z]) do df
    (default = mean(df.default, Weights(df.pmf)),)
  end
  
  default_df.default
  
  z_grid = unique(default_df.z)
  out = reshape(default_df.default, (length(a_grid), length(z_grid)))
  
  itp = interpolate((a_grid, z_grid), out, (Gridded(Linear()), Gridded(Constant())))
end

function default_probabilities(a_grid, exo_mc)
  size = (length(a_grid), length(exo_mc.state_values))
  
  default_probabilities(a_grid, exo_mc, falses(size), ones(size)/prod(size))
end


exo_MC = MarkovChain(z_MC, κ_MC, (:z, :κ))
exo_MC.state_values[1]

## Endogenous state
a_grid = LinRange(-10.0, 20.0, 300)

agg_state = AggStateNoFreshStart(0.05, a_grid, exo_MC)

@unpack value, policy, policies_full = solve_bellman(a_grid, exo_MC, (default=true, c=1.5), agg_state, maxiter=400 )
dist = stationary_distribution(exo_MC, a_grid, policy)
update_probabilities!(agg_state, a_grid, exo_MC, policies_full, dist)


using StatsPlots
plot(agg_state.default_choice)

plot(a_grid, value)
plot(a_grid, policy)


@df default_df plot(:a, :default, group=:z)