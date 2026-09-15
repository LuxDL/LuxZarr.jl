# LuxZarr.jl

Portable, chunked, and lazy [Zarr v3](https://zarr.dev/)-backed serialization for [Lux.jl](https://lux.csail.mit.edu/stable/) neural networks in Julia.

---

## Features

- **Chunked & Compressed**: Saves parameters and state trees into standard Zarr hierarchies (Zarr v3).
- **Lazy Loading**: Inspect and evaluate models directly from disk (`LazyLuxModel`) with minimal memory footprint, or eagerly `materialize` into standard arrays.
- **Architecture Metadata**: Serializes layer configurations to reconstruct models standalone without requiring the original model instance.
- **Extensible**: Simple two-method dispatch interface to support arbitrary custom layers and composite architectures.

---

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/LuxDL/LuxZarr.jl.git")
```

Once registered, you can install it directly with:

```julia
using Pkg
Pkg.add("LuxZarr")
```

---

## Quick Start

### 1. Save a Model

```julia
using Lux, LuxZarr, Random

rng = Random.default_rng()
model = Chain(Dense(10 => 32, relu), Dense(32 => 1))
ps, st = Lux.setup(rng, model)

# Save parameters, states, and architecture metadata
save_model("my_model.zarr", ps, st; model = model)
```

### 2. Load and Run

```julia
# Standalone Lazy Loading (reconstructs model architecture from metadata)
lazy_model = load_model("my_model.zarr")
x = rand(Float32, 10, 4)
y, _ = lazy_model(x)  # Evaluates seamlessly with on-disk parameters

# Guided Loading into a model definition
lazy_model = load_model("my_model.zarr", model)

# Eager Loading (returns concrete arrays)
ps, st, model = load_model("my_model.zarr"; lazy = false)

# Guided Loading into an existing (ps, st) skeleton
ps, st = load_model("my_model.zarr", ps, st; lazy = false)

# Materialize lazy parameters to memory (or GPU device)
ps_cpu = materialize(lazy_model.ps)
```

---

## Extending for Custom Layers

To enable metadata extraction and standalone reconstruction for your custom Lux layer, define methods for `extract_model_info` and `_reconstruct_layer`:

```julia
using Lux, LuxZarr

struct MyLinear{F} <: Lux.AbstractLuxLayer
    in_dims::Int
    out_dims::Int
    activation::F
end

# 1. Serialization hook: extract layer configuration
function LuxZarr.extract_model_info(layer::MyLinear)
    return Dict{String, Any}(
        "type" => "MyLinear",
        "in_dims" => layer.in_dims,
        "out_dims" => layer.out_dims,
        "activation" => string(layer.activation),
    )
end

# 2. Deserialization hook: reconstruct layer from metadata
function LuxZarr._reconstruct_layer(::Val{:MyLinear}, info::AbstractDict; kwargs...)
    act = get(info, "activation", "identity") == "relu" ? Lux.relu : identity
    return MyLinear(Int(info["in_dims"]), Int(info["out_dims"]), act)
end
```

---

## License

MIT License.
