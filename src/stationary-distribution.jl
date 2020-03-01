# """
#   Given a ordered vector [x_1 < x_2 < ... < x_N] and y ∈ [x_1, x_N] find i and λ such that y = λ x_i + (1-λ) x_{i+1}
# """
function findneighbours_line(vector, point)
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

function findneighbours(grids, point)
  if length(grids) == 1
    
    @unpack i_prev, i_next, λ = findneighbours_line(grids[1], point)
    
    return  [(i=i_prev, weight=λ),
             (i=i_next, weight=1-λ)]
  elseif length(grids) == 2
    #ngb_1 = findneighbours_line(endo.grids[1], point[1])
    #ngb_2 = findneighbours_line(endo.grids[2], point[2])
    
    ngb = [findneighbours_line(grids[i], point[i]) for i in 1:length(point)]
    #neighbours = NamedTuple{keys(point)}(Tuple(neighbours_vec))

    return [
     (i=(ngb[1].i_prev, ngb[2].i_prev), weight=ngb[1].λ * ngb[2].λ), 
     (i=(ngb[1].i_prev, ngb[2].i_next), weight=ngb[1].λ * (1-ngb[2].λ)),
     (i=(ngb[1].i_next, ngb[2].i_prev), weight=(1-ngb[1].λ) * ngb[2].λ),
     (i=(ngb[1].i_next, ngb[2].i_next), weight=(1-ngb[1].λ) * (1-ngb[2].λ))]

  else
    @error "not yet implemented for dim(endo) == $(length(grids))"
  end
end

@testset "findneighbours" begin
  grids = (a = LinRange(0.1,1,10), b=LinRange(1.1,2,10))

  point = (a=0.15, b=1.15)
  out = Aiyagari.findneighbours(grids, point)

  a_check = sum(grids.a[out[i].i[1]] .* out[i].weight for i in 1:length(out))
  b_check = sum(grids.b[out[i].i[2]] .* out[i].weight for i in 1:length(out)) 
  @test a_check ≈ point.a
  @test b_check ≈ point.b

  point = (a=0.11111, b=1.56788)
  out = Aiyagari.findneighbours(grids, point)

  a_check = sum(grids.a[out[i].i[1]] .* out[i].weight for i in 1:length(out))
  b_check = sum(grids.b[out[i].i[2]] .* out[i].weight for i in 1:length(out)) 
  @test a_check ≈ point.a
  @test b_check ≈ point.b
end

let a_grid = LinRange(1, 10, 100), p = 7.23456
  @unpack i_prev, i_next, λ = findneighbours_line(a_grid, p)

  a_prev, a_next = a_grid[[i_prev; i_next]]

  @testset "findneighbours" begin
    @test λ * a_prev + (1-λ) * a_next ≈ p
    @test a_prev < p < a_next
  end
end

## Build up transition matrix
function controlled_markov_chain(endo, exo, policy)
  lin_ind = LinearIndices((length(endo), length(exo)))

  len_exo = length(exo)
  len = length(endo) * len_exo
 
  n_endo_neighbours = 2 ^ dimension(endo) # 2 neighbours on line, 4 on plane, 8 in ℝ³
  
  len_sparse = len * len_exo * n_endo_neighbours

  I = zeros(Int, len_sparse)
  J = zeros(Int, len_sparse)
  V = zeros(len_sparse)

  controlled_markov_chain!(I, J, V, lin_ind, endo, exo, policy)
  
  (I=I, J=J, V=V, n=len)
end

function controlled_markov_chain!(I, J, V, lin_ind, endo, exo, policy)
  
  len_exo = length(exo)
  len_endo = length(endo)

  j = 0
  
  for i_exo in 1:len_exo
    # extract jump probabilities
    π = exo.mc.p
  
    for i_endo in 1:len_endo
      p = policy[i_endo, i_exo]
      ijump_mass_vec = findneighbours(endo.grids, p)
      for ijump_mass in ijump_mass_vec
        @unpack i, weight = ijump_mass 
        i_jump = linear_index(endo)[i...]
        mass = weight 
        for i_exo_next in 1:len_exo
          j += 1
          I[j] = lin_ind[i_endo, i_exo]
          J[j] = lin_ind[i_jump, i_exo_next]
          V[j] = mass * π[i_exo, i_exo_next]
        end
      end
    end
  end
end


## Stationary distribution

function stationary_distribution(endo, exo, policy)

  @unpack I, J, V, n = controlled_markov_chain(endo, exo, policy)
  
  transition = sparse(I, J, V, n, n)

  eigval, eigvec, _ = eigs(transpose(transition))
  is = findall(abs.(eigval) .≈ 1)
  @assert length(is) == 1
  @assert isreal(eigvec[:,is[1]]) 
  dist = real(eigvec[:,is[1]])
  
  dist ./= sum(dist)
  
  reshape(dist, size(policy))
end

# let value = value, policy=policy, z_mc = z_MC
#   lin_ind = LinearIndices(size(value))
# 
#   ngp_exo = length(z_mc.state_values)
#   n = length(policy)
#   len = n * ngp_exo * 2
# 
#   I = zeros(Int, len)
#   J = zeros(Int, len)
#   V = zeros(len)
# 
#   @btime Aiyagari.controlled_markov_chain!($I, $J, $V, $lin_ind, $z_mc, $a_grid, $policy)
# end