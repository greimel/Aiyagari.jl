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

function Aiyagari.iterate_bellman!(value_new, value_old, policy, policies_full, a_grid, z_mc, converged)
  r, β, γ = 0.05, 0.9, 0.9
  a_min, a_max = extrema(a_grid)
  
  for (i_z, nt) in enumerate(z_mc.state_values)
    @unpack z, κ = nt
    # Create interpolated expected value function
    exp_value = value_old * z_mc.p[i_z,:]

    itp_exp_value = interpolate(exp_value, BSpline(Cubic(Line(OnGrid()))))
    sitp_exp_value = scale(itp_exp_value, a_grid)
    
    for (i_a, a) in enumerate(a_grid) 
      
      obj = a_next -> begin
        c = a + z - κ - a_next/(1+r)
        u(c) + β * sitp_exp_value(a_next)
      end

      res = optimize(a_next -> - obj(a_next), a_min, a_max)
      
      a_no_def = Optim.minimizer(res)
      c_no_def = a + z - κ - a_no_def/(1+r)
      v_no_def = - Optim.minimum(res)
      
      # default
      c_default = (1-γ) * z
      a_default = (a + γ * z - κ) * (1+r)
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

exo_MC = MarkovChain(z_MC, κ_MC, (:z, :κ))
exo_MC.state_values[1]

## Endogenous state
a_grid = LinRange(-4.0, 4.0, 50)

@unpack value, policy, policies_full = solve_bellman(a_grid, exo_MC, (default=true, c=1.5), maxiter=400 )

using StructArrays, Plots

s = StructArray(policies_full)
plot(s.default)
plot(s.c)

using Plots
plot(a_grid, value)
plot(a_grid, policy)

