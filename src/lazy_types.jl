"""
    LazyParameters(data)

Lightweight transparent wrapper for lazy model parameter trees.
Forwards property access, iteration, keys, and indexing directly to the underlying `NamedTuple` or tree structure.
"""
struct LazyParameters{T}
    data::T
end

"""
    LazyState(data)

Lightweight transparent wrapper for lazy model state trees.
Forwards property access, iteration, keys, and indexing directly to the underlying `NamedTuple` or tree structure.
"""
struct LazyState{T}
    data::T
end

"""
    LazyLuxModel(model, ps, st)

Composite model object binding a model architecture with its lazy parameters and states.
"""
struct LazyLuxModel{M, P, S}
    model::M
    ps::P
    st::S
end

# Unwrapping helpers
unwrap(x::LazyParameters) = getfield(x, :data)
unwrap(x::LazyState) = getfield(x, :data)
unwrap(x) = x

Base.NamedTuple(x::LazyParameters) = NamedTuple(getfield(x, :data))
Base.NamedTuple(x::LazyState) = NamedTuple(getfield(x, :data))

# Property forwarding
Base.getproperty(x::Union{LazyParameters, LazyState}, sym::Symbol) = sym === :data ? getfield(x, :data) : getproperty(getfield(x, :data), sym)
Base.propertynames(x::Union{LazyParameters, LazyState}) = propertynames(getfield(x, :data))

Base.keys(x::Union{LazyParameters, LazyState}) = keys(getfield(x, :data))
Base.values(x::Union{LazyParameters, LazyState}) = values(getfield(x, :data))
Base.haskey(x::Union{LazyParameters, LazyState}, k) = haskey(getfield(x, :data), k)
Base.getindex(x::Union{LazyParameters, LazyState}, i) = getindex(getfield(x, :data), i)
Base.length(x::Union{LazyParameters, LazyState}) = length(getfield(x, :data))
Base.iterate(x::Union{LazyParameters, LazyState}, state...) = iterate(getfield(x, :data), state...)

# Functors integration
Functors.children(x::LazyParameters) = (data = getfield(x, :data),)
Functors.children(x::LazyState) = (data = getfield(x, :data),)
Functors.children(x::LazyLuxModel) = (model = getfield(x, :model), ps = getfield(x, :ps), st = getfield(x, :st))

# Adapt integration for transparent device transfers
Adapt.adapt_storage(to::MLDataDevices.CPUDevice, arr::Zarr.ZArray) = adapt(to, Array(arr))
Adapt.adapt_storage(to::MLDataDevices.AbstractDevice, arr::Zarr.ZArray) = adapt(to, Array(arr))
Adapt.adapt_storage(::Type{<:Array}, arr::Zarr.ZArray) = Array(arr)

Adapt.adapt_structure(to, x::LazyParameters) = LazyParameters(adapt(to, getfield(x, :data)))
Adapt.adapt_structure(to, x::LazyState) = LazyState(adapt(to, getfield(x, :data)))
Adapt.adapt_structure(to, x::LazyLuxModel) = LazyLuxModel(getfield(x, :model), adapt(to, getfield(x, :ps)), adapt(to, getfield(x, :st)))

# Iteration unpacking for (ps, st, model) = lazy_model
function Base.iterate(x::LazyLuxModel, state=1)
    if state == 1
        return (getfield(x, :ps), 2)
    elseif state == 2
        return (getfield(x, :st), 3)
    elseif state == 3
        return (getfield(x, :model), 4)
    else
        return nothing
    end
end
Base.length(::LazyLuxModel) = 3
Base.getindex(x::LazyLuxModel, i::Int) = i == 1 ? getfield(x, :ps) : (i == 2 ? getfield(x, :st) : (i == 3 ? getfield(x, :model) : throw(BoundsError(x, i))))

# Base.materialize implementations for lazy types
Base.materialize(arr::Zarr.ZArray) = Array(arr)
Base.materialize(lp::LazyParameters) = Functors.fmap(Base.materialize, unwrap(lp))
Base.materialize(ls::LazyState) = Functors.fmap(Base.materialize, unwrap(ls))
Base.materialize(lm::LazyLuxModel) = LazyLuxModel(lm.model, Base.materialize(lm.ps), Base.materialize(lm.st))

"""
    materialize(x; device=cpu_device())
    materialize(device, x)

Materialize lazy arrays in `x` into concrete in-memory or device-resident arrays.
Supports `Zarr.ZArray`, `LazyParameters`, `LazyState`, `LazyLuxModel`, and nested structures.
"""
materialize(arr::Zarr.ZArray; device=cpu_device()) = device(Array(arr))
materialize(lp::LazyParameters; device=cpu_device()) = device(Base.materialize(lp))
materialize(ls::LazyState; device=cpu_device()) = device(Base.materialize(ls))
materialize(lm::LazyLuxModel; device=cpu_device()) = LazyLuxModel(lm.model, materialize(lm.ps; device=device), materialize(lm.st; device=device))
materialize(dev::MLDataDevices.AbstractDevice, x) = materialize(x; device=dev)
materialize(x; device=cpu_device()) = device(Base.materialize(x))
