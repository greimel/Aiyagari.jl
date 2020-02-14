function get_optimum end

function solve_bellman!(value_old, value_new, policy, policies_full, a_grid, z_MC, converged, aggregate_state, params; maxiter=100, tol = √eps())
  
  for i in 1:maxiter
    iterate_bellman!(value_new, value_old, policy, policies_full, a_grid, z_MC, converged, aggregate_state, params)
    diff = norm(value_old - value_new)
    value_old .= value_new
    
    if diff < tol
      @info "converged after $i iterations, diff = $diff"
      break
    end
    if i % 200 == 0
      @info "it: $i, diff=$diff"
    end
    if i == maxiter
      @warn "reached $maxiter, diff= $diff"
    end
  end
end

function solve_bellman(a_grid, z_MC, proto_policy, proto_policy_full, aggregate_state, params; maxiter=200, tol=√eps())
  value_old = zeros(length(a_grid), length(z_MC.state_values))
  value_new = zeros(size(value_old))
  policy = fill(proto_policy, size(value_old))
  policies_full = fill(proto_policy_full, size(value_old))
  converged = trues(size(value_old))
  
  solve_bellman!(value_old, value_new, policy, policies_full, a_grid, z_MC, converged, aggregate_state, params; maxiter=maxiter, tol=tol)
  
  # checks
  # at_max = mean(policy .≈ a_grid[end])
  # at_min = mean(policy .≈ a_grid[1])
  # at_max > 0 && @warn "optimal policy is at upper bound $(100 * at_max) % of the time"
  # at_min > 0 && @warn "optimal policy is at lower bound $(100 * at_min) % of the time"

  all(converged) || @warn "optimization didn't converge at $(mean(converged) * 100)%"

  
  (value = value_new, policy = policy, policies_full=policies_full, converged=converged)
end

function iterate_bellman!(value_new, value_old, policy, policies_full, a_grid, z_mc, converged, agg_state, params)
  
  for (i_z, z) in enumerate(z_mc.state_values)
    # Create interpolated expected value function
    exp_value = value_old * z_mc.p[i_z,:]

    itp_exp_value = interpolate(exp_value, BSpline(Cubic(Line(OnGrid()))))

    𝔼V = extrapolate(
            scale(itp_exp_value, a_grid),
            Interpolations.Line()
            )
            
    
    for (i_a, a) in enumerate(a_grid) 
      states = (a=a, z=z)

      @unpack pol, pol_full, val, conv = get_optimum(states, agg_state, 𝔼V, params, a_grid)

      policy[i_a, i_z]    = pol 
      policies_full[i_a, i_z] = pol_full
      value_new[i_a, i_z] = val 
      converged[i_a, i_z] = conv 
    end
  end
end

