using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using LuxZarr
using ExplicitImports
print_explicit_imports(LuxZarr)
