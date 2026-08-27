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

# Functors integration
Functors.children(x::AbstractLazyTree) = (data = getfield(x, :data),)
Functors.children(x::LazyLuxModel) = (model = getfield(x, :model), ps = getfield(x, :ps), st = getfield(x, :st))

# Adapt integration for device transfers
Adapt.adapt_storage(to::MLDataDevices.CPUDevice, arr::Zarr.ZArray) = adapt(to, Array(arr))
Adapt.adapt_storage(to::MLDataDevices.AbstractDevice, arr::Zarr.ZArray) = adapt(to, Array(arr))
Adapt.adapt_storage(::Type{<:Array}, arr::Zarr.ZArray) = Array(arr)

Adapt.adapt_structure(to, x::T) where {T <: AbstractLazyTree} = T(adapt(to, getfield(x, :data)))
Adapt.adapt_structure(to, x::LazyLuxModel) = LazyLuxModel(getfield(x, :model), adapt(to, getfield(x, :ps)), adapt(to, getfield(x, :st)))

# Iteration & indexing for LazyLuxModel: (ps, st, model) = lazy_model
Base.iterate(x::LazyLuxModel, state=1) = state <= 3 ? (getfield(x, state), state + 1) : nothing
Base.length(::LazyLuxModel) = 3
Base.getindex(x::LazyLuxModel, i::Int) = 1 <= i <= 3 ? getfield(x, i) : throw(BoundsError(x, i))

# Base.materialize implementations
Base.materialize(arr::Zarr.ZArray) = Array(arr)
Base.materialize(x::AbstractLazyTree) = Functors.fmap(Base.materialize, unwrap(x))
Base.materialize(lm::LazyLuxModel) = LazyLuxModel(lm.model, Base.materialize(lm.ps), Base.materialize(lm.st))

"""
    materialize(x; device=cpu_device())
    materialize(device, x)

Materialize lazy arrays in `x` into concrete in-memory or device-resident arrays.
"""
materialize(arr::Zarr.ZArray; device=cpu_device()) = device(Array(arr))
materialize(x::AbstractLazyTree; device=cpu_device()) = device(Base.materialize(x))
materialize(lm::LazyLuxModel; device=cpu_device()) = LazyLuxModel(lm.model, materialize(lm.ps; device=device), materialize(lm.st; device=device))
materialize(dev::MLDataDevices.AbstractDevice, x) = materialize(x; device=dev)
materialize(x; device=cpu_device()) = device(Base.materialize(x))
