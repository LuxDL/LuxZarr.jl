abstract type AbstractLazyTree{T} end

"""
    LazyParameters(data)

Lightweight transparent wrapper for lazy model parameter trees.
"""
struct LazyParameters{T} <: AbstractLazyTree{T}
    data::T
end

"""
    LazyState(data)

Lightweight transparent wrapper for lazy model state trees.
"""
struct LazyState{T} <: AbstractLazyTree{T}
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

# Unwrapping and conversion helpers
unwrap(x::AbstractLazyTree) = getfield(x, :data)
unwrap(x) = x

Base.NamedTuple(x::AbstractLazyTree) = NamedTuple(getfield(x, :data))

# Transparent property, key, indexing, and iteration forwarding
Base.getproperty(x::AbstractLazyTree, sym::Symbol) = sym === :data ? getfield(x, :data) : getproperty(getfield(x, :data), sym)
Base.propertynames(x::AbstractLazyTree) = propertynames(getfield(x, :data))

Base.keys(x::AbstractLazyTree) = keys(getfield(x, :data))
Base.values(x::AbstractLazyTree) = values(getfield(x, :data))
Base.haskey(x::AbstractLazyTree, k) = haskey(getfield(x, :data), k)
Base.getindex(x::AbstractLazyTree, i) = getindex(getfield(x, :data), i)
Base.length(x::AbstractLazyTree) = length(getfield(x, :data))
Base.iterate(x::AbstractLazyTree, state...) = iterate(getfield(x, :data), state...)

# Functors and MLDataDevices integration
Functors.children(x::AbstractLazyTree) = (data = getfield(x, :data),)
Functors.children(x::LazyLuxModel) = (model = getfield(x, :model), ps = getfield(x, :ps), st = getfield(x, :st))

MLDataDevices.isleaf(::AbstractLazyTree) = true
MLDataDevices.isleaf(::LazyLuxModel) = true

_wrap_tree(::LazyParameters, data) = LazyParameters(data)
_wrap_tree(::LazyState, data) = LazyState(data)

# Adapt integration for device transfers
Adapt.adapt_structure(to, x::AbstractLazyTree) = _wrap_tree(x, adapt(to, materialize(x)))
Adapt.adapt_structure(to, x::LazyLuxModel) = LazyLuxModel(getfield(x, :model), adapt(to, getfield(x, :ps)), adapt(to, getfield(x, :st)))

# Iteration & indexing for LazyLuxModel: (ps, st, model) = lazy_model
Base.iterate(x::LazyLuxModel, state = 1) = state <= 3 ? (getfield(x, state), state + 1) : nothing
Base.length(::LazyLuxModel) = 3
Base.getindex(x::LazyLuxModel, i::Int) = 1 <= i <= 3 ? getfield(x, i) : throw(BoundsError(x, i))

"""
    materialize(x; device=nothing)
    materialize(device, x)

Materialize lazy Zarr arrays in `x` into concrete in-memory or device-resident arrays.
"""
function materialize end

_to_array(arr::Zarr.ZArray) = Array(arr)
_to_array(x) = x

_materialize_impl(arr::Zarr.ZArray) = Array(arr)
_materialize_impl(x::AbstractLazyTree) = Functors.fmap(_to_array, unwrap(x); exclude = _isleaf)
_materialize_impl(lm::LazyLuxModel) = LazyLuxModel(lm.model, _materialize_impl(lm.ps), _materialize_impl(lm.st))
_materialize_impl(x) = Functors.fmap(_to_array, x; exclude = _isleaf)

function materialize(x; device = nothing)
    m = _materialize_impl(x)
    return isnothing(device) ? m : device(m)
end

materialize(dev, x) = materialize(x; device = dev)
