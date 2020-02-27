using Documenter, Aiyagari

makedocs(;
    modules=[Aiyagari],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://gitlab.com/greimel/Aiyagari.jl/blob/{commit}{path}#L{line}",
    sitename="Aiyagari.jl",
    authors="Fabian Greimel <fabgrei@gmail.com>",
    assets=String[],
)
