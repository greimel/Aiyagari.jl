# Aiyagari.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://greimel.gitlab.io/Aiyagari.jl/dev)
[![Build Status](https://gitlab.com/greimel/Aiyagari.jl/badges/master/build.svg)](https://gitlab.com/greimel/Aiyagari.jl/pipelines)
[![Coverage](https://gitlab.com/greimel/Aiyagari.jl/badges/master/coverage.svg)](https://gitlab.com/greimel/Aiyagari.jl/commits/master)

This package is not registered. For installing the package, type

```julia
] add https://gitlab.com/greimel/Aiyagari.jl

using Aiyagari

```

in the julia REPL. For running the tests locally, you must use a patched version of `Literate.jl`,

```julia
] dev https://gitlab.com/greimel/Literate.jl

] test Aiyagari
```

For now, the package can solve the Huggett model and multiple version of macroeconomic models with infinitely-lived heterogenous agents with housing.

* only homeowners (with or without adjustment costs)
* homeowners and renters with size-separated markets properties to be rented and owned
