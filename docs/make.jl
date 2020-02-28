using Documenter, Aiyagari

PKG_HOME = joinpath(dirname(pathof(Aiyagari)), "..")

using LiterateWeave

TUTORIALS = joinpath(PKG_HOME, "docs", "src", "tutorials")
isdir(TUTORIALS) ? nothing : mkdir(TUTORIALS)

literateweave(joinpath(PKG_HOME, "examples/huggett.jl"), out_path=TUTORIALS, doctype="github")

makedocs(;
    modules=[Aiyagari],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
        "Huggett" => "tutorials/huggett.md"
    ],
    repo="https://gitlab.com/greimel/Aiyagari.jl/blob/{commit}{path}#L{line}",
    sitename="Aiyagari.jl",
    authors="Fabian Greimel <fabgrei@gmail.com>",
    assets=String[],
 )

