using Aiyagari
using Test

PKG_HOME = joinpath(dirname(pathof(Aiyagari)), "..")

using Literate

GENERATED = joinpath(PKG_HOME, "test", "generated")
EXMPL = joinpath(PKG_HOME, "examples")

isdir(GENERATED) ? nothing : mkdir(GENERATED)

map(Aiyagari.examples) do exmpl
  Literate.script(joinpath(EXMPL, exmpl*".jl"), GENERATED)

  @testset "$exmpl regression test" begin
    include(joinpath(GENERATED, exmpl*".jl"))
  end
end

