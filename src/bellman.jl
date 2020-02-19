function get_optimum end

function solve_bellman(a_grid, z_MC, aggregate_state, params, hh::OwnOrRent; maxiter=200, tol=eps()^0.4)
  value_old = zeros(length(a_grid), length(z_MC.state_values))
  value_new = zeros(size(value_old))
  
  proto_own  = proto_policy(a_grid, z_MC, value_new, aggregate_state, params, Owner())
  proto_rent = proto_policy(a_grid, z_MC, value_new, aggregate_state, params, Renter())
  
  proto_pol = [proto_own.proto_pol, proto_rent.proto_pol]
  proto_pol_full = [proto_own.proto_pol_full, proto_rent.proto_pol_full]
  
  policy = fill.(proto_pol, Ref(size(value_old)))
  policies_full = fill.(proto_pol_full, Ref(size(value_old)))
  converged = [trues(size(value_old)), trues(size(value_old))]
  
  solve_bellman!(value_old, value_new, policy, policies_full, a_grid, z_MC, converged, aggregate_state, params, hh::Household; maxiter=maxiter, tol=tol)
  
  # checks
  # at_max = mean(policy .≈ a_grid[end])
  # at_min = mean(policy .≈ a_grid[1])
  # at_max > 0 && @warn "optimal policy is at upper bound $(100 * at_max) % of the time"
  # at_min > 0 && @warn "optimal policy is at lower bound $(100 * at_min) % of the time"

  all(all.(converged)) || @warn "optimization didn't converge at $(mean.(converged) * 100)%"

  
  (val = value_new, policy = policy, policies_full=StructArray.(policies_full), converged=converged)
end

function solve_bellman!(value_old, value_new, policy, policies_full, a_grid, z_MC, converged, aggregate_state, params, hh::OwnOrRent; maxiter=100, tol = √eps())
  
  value_own = zeros(size(value_old))
  value_rent = zeros(size(value_old))
  
  prog = ProgressThresh(tol, "Solving Bellman equation")
  for i in 1:maxiter
    # own
    iterate_bellman!(value_own, value_old, policy[1], policies_full[1], a_grid, z_MC, converged[1], aggregate_state, params, Owner())
    # rent
    iterate_bellman!(value_rent, value_old, policy[2], policies_full[2], a_grid, z_MC, converged[2], aggregate_state, params, Renter())
    
    value_new .= max.(value_own, value_rent)
    
    diff = norm(value_old - value_new)
    ProgressMeter.update!(prog, diff)
    value_old .= value_new
    
    if diff < tol
      break
    end
    if i == maxiter
      print("\n"^2)
      @warn "reached $maxiter, diff= $diff"
    end
  end
end

function solve_bellman!(value_old, value_new, policy, policies_full, a_grid, z_MC, converged, aggregate_state, params, hh::Household; maxiter=100, tol = √eps())
  
  prog = ProgressThresh(tol, "Solving Bellman equation")
  for i in 1:maxiter
    iterate_bellman!(value_new, value_old, policy, policies_full, a_grid, z_MC, converged, aggregate_state, params, hh::Household)
    diff = norm(value_old - value_new)
    ProgressMeter.update!(prog, diff)
    value_old .= value_new
    
    if diff < tol
      break
    end
    if i == maxiter
      print("\n"^2)
      @warn "reached $maxiter, diff= $diff"
    end
  end
end

function solve_bellman(a_grid, z_MC, aggregate_state, params, hh::Household; maxiter=200, tol=eps()^0.4)
  value_old = zeros(length(a_grid), length(z_MC.state_values))
  value_new = zeros(size(value_old))
  
  @unpack proto_pol, proto_pol_full = proto_policy(a_grid, z_MC, value_new, aggregate_state, params, hh)
  
  policy = fill(proto_pol, size(value_old))
  policies_full = fill(proto_pol_full, size(value_old))
  converged = trues(size(value_old))
  
  solve_bellman!(value_old, value_new, policy, policies_full, a_grid, z_MC, converged, aggregate_state, params, hh::Household; maxiter=maxiter, tol=tol)
  
  # checks
  # at_max = mean(policy .≈ a_grid[end])
  # at_min = mean(policy .≈ a_grid[1])
  # at_max > 0 && @warn "optimal policy is at upper bound $(100 * at_max) % of the time"
  # at_min > 0 && @warn "optimal policy is at lower bound $(100 * at_min) % of the time"

  all(converged) || @warn "optimization didn't converge at $(mean(converged) * 100)%"

  
  (val = value_new, policy = policy, policies_full=StructArray(policies_full), converged=converged)
end

function proto_policy(a_grid, z_MC, value, agg_state, params, hh::Household)
  
  states = (a=a_grid[1], z=z_MC.state_values[1])
  𝔼V = expected_value2(value, z_MC.p[1,:], BSpline(Linear()), a_grid)
  @unpack pol, pol_full = get_optimum(states, agg_state, 𝔼V, params, a_grid, hh::Household)
        
  (proto_pol=pol, proto_pol_full=pol_full)
end


function expected_value2(value, π, itp_scheme, a_grid)
  #itp_scheme = BSpline(Cubic(Line(OnGrid())))
  #itp_scheme = BSpline(Linear())

  exp_value = value * π
    
  itp_exp_value = interpolate(exp_value, itp_scheme)

  𝔼V = extrapolate(
          scale(itp_exp_value, a_grid),
          Interpolations.Line()
          )
end
        
function iterate_bellman!(value_new, value_old, policy, policies_full, a_grid, z_mc, converged, agg_state, params, hh::Household)
  n = length(value_new)
  prog = Progress(n, desc="Iterating", offset=1, dt=1)
  for (i_z, z) in enumerate(z_mc.state_values)
    # Create interpolated expected value function
    #𝔼V = expected_value2(value_old, z_mc.p[i_z,:], BSpline(Linear()), a_grid)
    𝔼V = expected_value2(value_old, z_mc.p[i_z,:],  BSpline(Cubic(Line(OnGrid()))), a_grid)
    
    for (i_a, a) in enumerate(a_grid) 
      states = (a=a, z=z)
      ProgressMeter.next!(prog)
      @unpack pol, pol_full, val, conv = get_optimum(states, agg_state, 𝔼V, params, a_grid, hh::Household)

      policy[i_a, i_z]    = pol 
      policies_full[i_a, i_z] = pol_full
      value_new[i_a, i_z] = val 
      converged[i_a, i_z] = conv 
    end
  end
end

