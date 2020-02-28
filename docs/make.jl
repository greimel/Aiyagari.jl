using Documenter, Aiyagari

PKG_HOME = joinpath(dirname(pathof(Aiyagari)), "..")

using LiterateWeave

EXMPL = joinpath(PKG_HOME, "examples")
TUTORIALS = joinpath(PKG_HOME, "docs", "src", "tutorials")
isdir(TUTORIALS) ? nothing : mkdir(TUTORIALS)

map(Aiyagari.examples) do exmpl
  literateweave(joinpath(EXMPL, exmpl*".jl"), out_path=TUTORIALS, doctype="github")
end

examples_pairs = [exmpl => "tutorials/" * exmpl*".md" for exmpl in Aiyagari.examples]

makedocs(;
    modules=[Aiyagari],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md";
        examples_pairs
    ],
    repo="https://gitlab.com/greimel/Aiyagari.jl/blob/{commit}{path}#L{line}",
    sitename="Aiyagari.jl",
    authors="Fabian Greimel <fabgrei@gmail.com>",
    assets=String[],
 )

