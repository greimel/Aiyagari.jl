function get_optimum end
function get_optimum_last end

############################################################
## Helpers for pre-allocating objects
############################################################

function proto_policy(endo, exo, value, agg_state, params, hh::Household)
  mc = MarkovChain(exo)
  
  state = merge(endo.grid[1], exo.grid[1])
  𝔼V = extrapolated_𝔼V(endo, BSpline(Linear()), value, exo, 1, 𝔼(hh))
  @unpack pol, pol_full = get_optimum(state, agg_state, 𝔼V, params, endo, hh::Household)
        
  (proto_pol=pol, proto_pol_full=pol_full)
end

############################################################
## Iterating the bellman operator etc
############################################################

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
      @unpack pol, pol_full, val, conv = get_optimum(states, agg_state, 𝔼V, params, endo, hh)

      policy[i_endo, i_exo]    = pol 
      policies_full[i_endo, i_exo] = pol_full
      value_new[i_endo, i_exo] = val 
      converged[i_endo, i_exo] = conv 
    end
  end
end

############################################################
## Finite Horizon (OLG)
############################################################

function solve_bellman(endo, exo, aggregate_state, params, hh::Household, t_grid)
  containers = initialize_values_policies(endo, exo, aggregate_state, params, hh, t_grid)
  @unpack value, policy, policies_full, converged = containers
  
  solve_bellman_T!(value, policy, policies_full, endo, exo, converged, aggregate_state, params, hh, t_grid)
      
  (val = value, policy = policy, policies_full=StructArray(policies_full), converged=converged)
end

function initialize_values_policies(endo, exo, aggregate_state, params, hh::Household, t_grid)
  container_size = (length(endo), length(exo), length(t_grid))

  value = zeros(container_size)
  
  @unpack proto_pol, proto_pol_full = proto_policy(endo, exo, value[:,:,1], aggregate_state, params, hh)
  
  policy = fill(proto_pol, container_size)
  policies_full = fill(proto_pol_full, container_size)
  converged = trues(container_size)
  
  (value=value, policy=policy, policies_full=policies_full, converged=converged)
end

function solve_bellman_T!(value, policy, policies_full, endo, exo, converged, aggregate_state, params, hh::Household, t_grid)
  
  # Solving the last period (special-cased for bequests, etc)
  iterate_bellman_last!(@view(value[:,:,end]),
                   @view(policy[:,:,end]),
                   @view(policies_full[:,:,end]),
                   endo, exo,
                   @view(converged[:,:,end]), aggregate_state, params, hh)
  
  # Solving all other periods
  @showprogress "Bellman(T): VFI" for i_t in reverse(1:length(t_grid)-1)
    iterate_bellman!(@view(value[:,:,i_t]),
                     @view(value[:,:,i_t+1]),
                     @view(policy[:,:,i_t]),
                     @view(policies_full[:,:,i_t]),
                     endo, exo,
                     @view(converged[:,:,i_t]), aggregate_state, params, hh)
  end
  
  number_conv = sum(converged)
  
  length(converged) == number_conv || @warn "Bellman didn't converge at $(round((1-number_conv / length(converged)) * 100, digits=4))% ($(length(converged) - number_conv) states)"
end

# For OLG: solve last period
function iterate_bellman_last!(value, policy, policies_full, endo, exo, converged, agg_state, params, hh::Household)
  mc = MarkovChain(exo)
  n = length(value)
  prog = Progress(n, desc="Iterating", offset=1, dt=1)
  for (i_exo, exo_state) in enumerate(mc.state_values)
    # Create interpolated expected value function
  
    for (i_endo, endo_state) in enumerate(endo.grid) 
      states = merge(endo_state, exo_state)
      ProgressMeter.next!(prog)
      @unpack pol, pol_full, val, conv = get_optimum_last(states, agg_state, params, endo, hh)

      policy[i_endo, i_exo]    = pol 
      policies_full[i_endo, i_exo] = pol_full
      value[i_endo, i_exo] = val 
      converged[i_endo, i_exo] = conv 
    end
  end
end

############################################################
## Infinite horizon
############################################################

function solve_bellman(endo, exo, aggregate_state, params, hh::Household; maxiter=200, rtol=eps()^0.4)
  containers = initialize_values_policies(endo, exo, aggregate_state, params, hh)
  @unpack value_old, value_new, policy, policies_full, converged = containers
  
  solve_bellman!(value_old, value_new, policy, policies_full, endo, exo, converged, aggregate_state, params, hh::Household; maxiter=maxiter, rtol=rtol)
    
  (val = value_new, policy = policy, policies_full=StructArray(policies_full), converged=converged)
end

function initialize_values_policies(endo, exo, aggregate_state, params, hh::Household)
  container_size = (length(endo), length(exo))

  value_old = zeros(container_size)
  value_new = zeros(container_size)
  
  @unpack proto_pol, proto_pol_full = proto_policy(endo, exo, value_new, aggregate_state, params, hh)
  
  policy = fill(proto_pol, container_size)
  policies_full = fill(proto_pol_full, container_size)
  converged = trues(container_size)
  
  (value_old=value_old, value_new=value_new, policy=policy, policies_full=policies_full, converged=converged)
end

function solve_bellman!(value_old, value_new, policy, policies_full, endo, exo, converged, aggregate_state, params, hh::Household; maxiter, rtol)
  
  prog = ProgressThresh(rtol, "Bellman: VFI")
  for i in 1:maxiter
    iterate_bellman!(value_new, value_old, policy, policies_full, endo, exo, converged, aggregate_state, params, hh)
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
  number_conv = sum(converged)
  
  length(converged) == number_conv || @warn "Bellman didn't converge at $(round((1-number_conv / length(converged)) * 100, digits=4))% ($(length(converged) - number_conv) states)"

end

############################################################
## Special cases for coupled value functions
############################################################

function solve_bellman(endo, exo, aggregate_state, params, chh::CoupledHouseholds; maxiter=200, rtol=eps()^0.4)
  containers = initialize_values_policies(endo, exo, aggregate_state, params, chh)
  @unpack W_old, W_new, V, policy, policies_full, policy_hh, converged = containers
  
  solve_bellman!(W_old, W_new, V, policy, policies_full, policy_hh, endo, exo, converged, aggregate_state, params, chh; maxiter=maxiter, rtol=rtol)
  
  (val = W_new, policy = policy, policy_hh=policy_hh, policies_full=StructArray.(policies_full), converged=converged)
end

function initialize_values_policies(endo, exo, aggregate_state, params, chh::CoupledHouseholds)
  container_size = (length(endo), length(exo))
  
  hh_tup = households(chh)
  n = length(chh)
  
  V         = [zeros(container_size) for i in 1:n]
  W_old     = [zeros(container_size) for i in 1:n]
  W_new     = [zeros(container_size) for i in 1:n]
  converged = [trues(container_size) for i in 1:n]
  
  policy_hh = [zeros(Int, container_size) for i in 1:n]
  
    
  # @unpack proto_pol, proto_pol_full
  proto = map(1:n) do i
    proto_policy(endo, exo, V[1], aggregate_state, params[i], hh_tup[i])
  end
    
  proto_pol = [p.proto_pol for p in proto]
  proto_pol_full = [p.proto_pol_full for p in proto]
  
  policy = fill.(proto_pol, Ref(container_size))
  policies_full = fill.(proto_pol_full, Ref(container_size))
  
  (W_old=W_old, W_new=W_new, V=V, policy=policy, policies_full=policies_full, policy_hh=policy_hh, converged=converged)
end

function solve_bellman!(W_old::Vector, W_new::Vector, V::Vector, policy::Vector, policies_full::Vector, policies_hh::Vector, endo, exo, converged::Vector, aggregate_state, params::Vector, chh::CoupledHouseholds; maxiter, rtol)
    
  prog = ProgressThresh(rtol, "Solving Bellman equation")

  for i in 1:maxiter
    iterate_bellman!.(V, W_old, policy, policies_full, Ref(endo), Ref(exo), converged, Ref(aggregate_state), params, households(chh))
    
    update_coupled_values!(W_new, V, policies_hh)
    
    diff = norm.(W_old .- W_new)
    
    adj_fact = max.(norm.(W_old), norm.(W_new))
    
    relative_error = maximum(diff ./ adj_fact)
    
    ProgressMeter.update!(prog, relative_error)
    
    for i in 1:length(W_old)
      W_old[i] .= W_new[i]
    end
    
    if relative_error < rtol
      break
    end

    if i == maxiter
      print("\n"^2)
      @warn "reached $maxiter, diff= $relative_error"
    end
  end
  
  all(all.(converged)) || @warn "optimization didn't converge at $((1 .- mean.(converged)) .* 100)%"

end

#function update_coupled_values! end # needs to be provided

function update_coupled_values!(W, V, policies_hh)
  W_own, W_rent = W
  V_own, V_rent = V
  policies_own, policies_rent = policies_hh
  
  W_own  .= max.(V_own, V_rent)
  policies_own .= V_own .> V_rent
  
  W_rent .= max.(V_own, V_rent)
  policies_rent .= V_own .> V_rent
  
  nothing
end

