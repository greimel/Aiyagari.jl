function get_optimum end

function solve_bellman(endo, exo, aggregate_state, params, hh::OwnOrRent; maxiter=200, rtol=eps()^0.4)
  value_old = zeros(length(endo), length(exo))
  value_new = zeros(size(value_old))
  owner = trues(size(value_old))
    
  proto_own  = proto_policy(endo, exo, value_new, aggregate_state, params[1], hh.owner)
  proto_rent = proto_policy(endo, exo, value_new, aggregate_state, params[2], hh.renter)
  
  proto_pol = [proto_own.proto_pol, proto_rent.proto_pol]
  proto_pol_full = [proto_own.proto_pol_full, proto_rent.proto_pol_full]
  
  policy = fill.(proto_pol, Ref(size(value_old)))
  policies_full = fill.(proto_pol_full, Ref(size(value_old)))
  converged = [trues(size(value_old)), trues(size(value_old))]
  
  solve_bellman!(value_old, value_new, policy, policies_full, owner, endo, exo, converged, aggregate_state, params, hh::Household; maxiter=maxiter, rtol=rtol)
  
  # checks
  # at_max = mean(policy .≈ a_grid[end])
  # at_min = mean(policy .≈ a_grid[1])
  # at_max > 0 && @warn "optimal policy is at upper bound $(100 * at_max) % of the time"
  # at_min > 0 && @warn "optimal policy is at lower bound $(100 * at_min) % of the time"

  all(all.(converged)) || @warn "optimization didn't converge at $(mean.(converged) * 100)%"

  
  (val = value_new, policy = policy, owner=owner, policies_full=StructArray.(policies_full), converged=converged)
end

function solve_bellman!(value_old, value_new, policy, policies_full, owner, endo, exo, converged, aggregate_state, params, hh::OwnOrRent; maxiter=100, rtol = √eps())
  
  value_own = zeros(size(value_old))
  value_rent = zeros(size(value_old))
  
  prog = ProgressThresh(rtol, "Solving Bellman equation")
  for i in 1:maxiter
    # own
    iterate_bellman!(value_own, value_old, policy[1], policies_full[1], endo, exo, converged[1], aggregate_state, params[1], hh.owner)

    # rent
    iterate_bellman!(value_rent, value_old, policy[2], policies_full[2], endo, exo, converged[2], aggregate_state, params[2], hh.renter)
    
    owner .= value_own .> value_rent
    value_new .= max.(value_own, value_rent)
    
    diff = norm(value_old - value_new)
    
    adj_fact = max(norm(value_old), norm(value_new))
     
    ProgressMeter.update!(prog, diff / adj_fact)
    value_old .= value_new
    
    if diff < rtol * adj_fact
      break
    end

    if i == maxiter
      print("\n"^2)
      @warn "reached $maxiter, diff= $diff"
    end
  end
end

function solve_bellman!(value_old, value_new, policy, policies_full, endo, exo, converged, aggregate_state, params, hh::Household; maxiter=100, rtol = √eps())
  
  prog = ProgressThresh(rtol, "Solving Bellman equation")
  for i in 1:maxiter
    iterate_bellman!(value_new, value_old, policy, policies_full, endo, exo, converged, aggregate_state, params, hh::Household)
    diff = norm(value_old - value_new)
    
    adj_fact = max(norm(value_old), norm(value_new))
     
    ProgressMeter.update!(prog, diff / adj_fact)
    value_old .= value_new
    
    if diff < rtol * adj_fact
      break
    end
    if i == maxiter
      print("\n"^2)
      @warn "reached $maxiter, diff= $diff"
    end
  end
end

function solve_bellman(endo, exo, aggregate_state, params, hh::Household; maxiter=200, rtol=eps()^0.4)
  value_old = zeros(length(endo), length(exo))
  value_new = zeros(size(value_old))
  
  @unpack proto_pol, proto_pol_full = proto_policy(endo, exo, value_new, aggregate_state, params, hh)
  
  policy = fill(proto_pol, size(value_old))
  policies_full = fill(proto_pol_full, size(value_old))
  converged = trues(size(value_old))
  
  solve_bellman!(value_old, value_new, policy, policies_full, endo, exo, converged, aggregate_state, params, hh::Household; maxiter=maxiter, rtol=rtol)
  
  # checks
  # at_max = mean(policy .≈ a_grid[end])
  # at_min = mean(policy .≈ a_grid[1])
  # at_max > 0 && @warn "optimal policy is at upper bound $(100 * at_max) % of the time"
  # at_min > 0 && @warn "optimal policy is at lower bound $(100 * at_min) % of the time"
  
  number_conv = sum(converged)
  
  length(converged) == number_conv || @warn "Bellman didn't converge at $(round((1-number_conv / length(converged)) * 100, digits=4))% ($(length(converged) - number_conv) states)"

  
  (val = value_new, policy = policy, policies_full=StructArray(policies_full), converged=converged)
end

function proto_policy(endo, exo, value, agg_state, params, hh::Household)
  mc = MarkovChain(exo)
  
  state = merge(endo.grid[1], exo.grid[1])
  𝔼V = extrapolated_𝔼V(endo, BSpline(Linear()), value, exo, 1, 𝔼(hh))
  @unpack pol, pol_full = get_optimum(state, agg_state, 𝔼V, params, endo, hh::Household)
        
  (proto_pol=pol, proto_pol_full=pol_full)
end

function iterate_bellman!(value_new, value_old, policy, policies_full, endo, exo, converged, agg_state, params, hh::Household)
  mc = MarkovChain(exo)
  n = length(value_new)
  prog = Progress(n, desc="Iterating", offset=1, dt=1)
  for (i_exo, exo_state) in enumerate(mc.state_values)
    # Create interpolated expected value function
    itp_scheme = BSpline(Cubic(Line(OnGrid())))
    𝔼V = extrapolated_𝔼V(endo, itp_scheme, value_old, exo, i_exo, 𝔼(hh))
    
    for (i_endo, endo_state) in enumerate(endo.grid) 
      states = merge(endo_state, exo_state)
      ProgressMeter.next!(prog)
      @unpack pol, pol_full, val, conv = get_optimum(states, agg_state, 𝔼V, params, endo, hh::Household)

      policy[i_endo, i_exo]    = pol 
      policies_full[i_endo, i_exo] = pol_full
      value_new[i_endo, i_exo] = val 
      converged[i_endo, i_exo] = conv 
    end
  end
end

