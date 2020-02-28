using Aiyagari
using Test

PKG_HOME = joinpath(dirname(pathof(Aiyagari)), "..")

using Literate

GENERATED = joinpath(PKG_HOME, "test", "generated")
isdir(GENERATED) ? nothing : mkdir(GENERATED)

Literate.script(joinpath(PKG_HOME, "examples/huggett.jl"), GENERATED )

@testset "Huggett regression test" begin
  include(joinpath(GENERATED, "huggett.jl"))
end