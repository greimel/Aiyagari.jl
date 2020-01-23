"""
  Given a ordered vector [x_1 < x_2 < ... < x_N] and y ∈ [x_1, x_N] find i and λ such that y = λ x_i + (1-λ) x_{i+1}
"""
function findneighbours(vector, point)
  i_next = searchsortedfirst(vector, point)
  
  if i_next == 1
    i_prev = 1
    λ = 1.0
  else
    i_prev = i_next - 1
    prev = vector[i_prev]
    next = vector[i_next]

    Δ = next - prev
    λ = (next - point) / Δ
  end

  (i_prev=i_prev, i_next=i_next, λ=λ)
end


using Test

let a_grid = LinRange(1, 10, 100), p = 7.23456
  @unpack i_prev, i_next, λ = findneighbours(a_grid, p)

  a_prev, a_next = a_grid[[i_prev; i_next]]

  @testset "findneighbours" begin
    @test λ * a_prev + (1-λ) * a_next ≈ p
    @test a_prev < p < a_next
  end
end

## Build up transition matrix
function controlled_markov_chain(z_mc, a_grid, policy)
  lin_ind = LinearIndices(size(policy))

  ngp_exo = length(z_mc.state_values)
  n = length(policy)
  len = n * ngp_exo * 2

  I = zeros(Int, len)
  J = zeros(Int, len)
  V = zeros(len)

  controlled_markov_chain!(I, J, V, lin_ind, z_mc, a_grid, policy)
  
  (I=I, J=J, V=V, n=n)
end

function controlled_markov_chain!(I, J, V, lin_ind, z_mc, a_grid, policy)
  
  ngp_exo = length(z_mc.state_values)
  n = length(policy)
  len = n * ngp_exo * 2
  j = 0
  
  for (i_z, z) in enumerate(z_mc.state_values)
    # extract jump probabilities
    π = z_mc.p
  
    for (i_a, a) in enumerate(a_grid) 
      p = policy[i_a, i_z]
      @unpack i_prev, i_next, λ = findneighbours(a_grid, p)
  
      for i_z_next in 1:ngp_exo
        j += 1
        I[j] = lin_ind[i_a, i_z]
        J[j] = lin_ind[i_prev, i_z_next]
        V[j] = λ * π[i_z, i_z_next]
      end
      for i_z_next in 1:ngp_exo
        j += 1
        I[j] = lin_ind[i_a, i_z]
        J[j] = lin_ind[i_next, i_z_next]
        V[j] = (1-λ) * π[i_z, i_z_next]
      end
    end
  end
end


## Stationary distribution

function stationary_distribution(z_mc, a_grid, policy)

  @unpack I, J, V, n = controlled_markov_chain(z_mc, a_grid, policy)
  
  transition = sparse(I, J, V, n, n)

  eigval, eigvec, _ = eigs(transpose(transition))
  is = findall(abs.(eigval) .≈ 1)
  @assert length(is) == 1
  @assert isreal(eigvec[:,is[1]]) 
  dist = real(eigvec[:,is[1]])
  
  dist ./= sum(dist)
end

