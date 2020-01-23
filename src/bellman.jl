function iterate_bellman! end

function solve_bellman!(value_old, value_new, policy, a_grid, z_MC, converged; maxiter=100, tol = √eps())
  
  for i in 1:maxiter
    iterate_bellman!(value_new, value_old, policy, a_grid, z_MC, converged)
    diff = norm(value_old - value_new)
    value_old .= value_new
    
    if diff < tol
      @info "converged after $i iterations, diff = $diff"
      break
    end
    if i == 50
      @warn "reached $maxiter, diff= $diff"
    end
  end
end

function solve_bellman(a_grid, z_MC; maxiter=200, tol=√eps())
  value_old = zeros(length(a_grid), length(z_MC.state_values))
  value_new = zeros(size(value_old))
  policy = zeros(size(value_old))
  converged = trues(size(value_old))

  solve_bellman!(value_old, value_new, policy, a_grid, z_MC, converged; maxiter=maxiter, tol=tol)
  
  # checks
  at_max = mean(policy .≈ a_grid[end])
  at_min = mean(policy .≈ a_grid[1])
  at_max > 0 && @warn "optimal policy is at upper bound $(100 * at_max) % of the time"
  at_min > 0 && @warn "optimal policy is at lower bound $(100 * at_min) % of the time"

  all(converged) || @warn "optimization didn't converge at $(mean(converged) * 100)%"

  
  (value = value_new, policy = policy, converged=converged)
end