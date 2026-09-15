module LuxExt

using LuxZarr: LuxZarr,
    _keypath_to_path, _deserialize_scalar, _get_zarr_node, _isleaf, _guided_load_tree,
    LazyParameters, LazyState, LazyLuxModel, unwrap, materialize
import LuxZarr: extract_model_info, reconstruct_model_from_info, load_model, reconstruct_layer, _setup_model_skeleton
using Lux: Lux, AbstractLuxLayer, Dense, Conv, Chain, Parallel, BranchLayer, BatchNorm, SkipConnection,
    WrappedFunction, Dropout, NoOpLayer
using LuxCore: LuxCore
using Functors: Functors
using Adapt: adapt
using Random: Random, default_rng
using Zarr: Zarr

# Hook implementations via type-based dispatch
LuxZarr._get_lux_version(::Val{:Lux}) = string(pkgversion(Lux))
LuxZarr._get_lux_version(::AbstractLuxLayer) = string(pkgversion(Lux))

function LuxZarr.load_model(
        store_or_path,
        model::Union{AbstractLuxLayer, NamedTuple};
        lazy::Bool = true,
        rng = default_rng(),
        kwargs...,
    )
    ps, st = if model isa NamedTuple
        nt_setups = map(m -> Lux.setup(rng, m), model)
        (map(first, nt_setups), map(last, nt_setups))
    else
        Lux.setup(rng, model)
    end
    ps_loaded, st_loaded = LuxZarr.load_model(store_or_path, ps, st; lazy = lazy, kwargs...)
    if lazy
        return LazyLuxModel(model, ps_loaded, st_loaded)
    else
        return (unwrap(ps_loaded), unwrap(st_loaded))
    end
end

# Callable LazyLuxModel
(lm::LazyLuxModel)(x, ps = lm.ps, st = lm.st) = lm.model(x, materialize(ps), materialize(st))

# Transparent application of AbstractLuxLayer with LazyParameters / LazyState
(l::LuxCore.AbstractLuxLayer)(x, ps::LazyParameters, st::LazyState) = l(x, materialize(ps), materialize(st))
(l::LuxCore.AbstractLuxLayer)(x, ps::LazyParameters, st = NamedTuple()) = l(x, materialize(ps), materialize(st))
(l::LuxCore.AbstractLuxLayer)(x, ps, st::LazyState) = l(x, materialize(ps), materialize(st))

(l::LuxCore.AbstractLuxWrapperLayer)(x, ps::LazyParameters, st::LazyState) = l(x, materialize(ps), materialize(st))
(l::LuxCore.AbstractLuxWrapperLayer)(x, ps::LazyParameters, st = NamedTuple()) = l(x, materialize(ps), materialize(st))
(l::LuxCore.AbstractLuxWrapperLayer)(x, ps, st::LazyState) = l(x, materialize(ps), materialize(st))

# Model metadata extraction via multiple dispatch
function LuxZarr.extract_model_info(model::AbstractLuxLayer)
    if hasproperty(model, :layers)
        return Dict{String, Any}(
            "type" => string(nameof(typeof(model))),
            "summary" => string(model),
            "layers" => [extract_model_info(l) for l in _get_layer_list(model.layers)],
        )
    end
    return Dict{String, Any}(
        "type" => string(nameof(typeof(model))),
        "summary" => string(model),
    )
end

function LuxZarr.extract_model_info(model::Lux.Dense)
    return Dict{String, Any}(
        "type" => "Dense",
        "summary" => string(model),
        "in_dims" => model.in_dims,
        "out_dims" => model.out_dims,
        "use_bias" => Lux.has_bias(model),
        "activation" => string(model.activation),
    )
end

function LuxZarr.extract_model_info(model::Lux.Conv)
    return Dict{String, Any}(
        "type" => "Conv",
        "summary" => string(model),
        "in_chs" => model.in_chs,
        "out_chs" => model.out_chs,
        "kernel_size" => collect(model.kernel_size),
        "activation" => string(model.activation),
        "use_bias" => Lux.has_bias(model),
    )
end

function LuxZarr.extract_model_info(model::Lux.BatchNorm)
    return Dict{String, Any}(
        "type" => "BatchNorm",
        "summary" => string(model),
        "chs" => model.chs,
        "activation" => string(model.activation),
        "epsilon" => Float64(model.epsilon),
        "momentum" => Float64(model.momentum),
        "affine" => Bool(model.affine),
        "track_stats" => Bool(model.track_stats),
    )
end

function LuxZarr.extract_model_info(model::Lux.Parallel)
    return Dict{String, Any}(
        "type" => "Parallel",
        "summary" => string(model),
        "connection" => string(model.connection),
        "layers" => [extract_model_info(l) for l in _get_layer_list(model.layers)],
    )
end

function LuxZarr.extract_model_info(model::Lux.SkipConnection)
    return Dict{String, Any}(
        "type" => "SkipConnection",
        "summary" => string(model),
        "layers" => extract_model_info(model.layers),
        "connection" => string(model.connection),
    )
end

function LuxZarr.extract_model_info(model::Lux.WrappedFunction)
    return Dict{String, Any}(
        "type" => "WrappedFunction",
        "summary" => string(model),
        "func" => string(model.func),
    )
end

function LuxZarr.extract_model_info(model::Lux.Dropout)
    dims_info = if model.dims isa Colon
        ":"
    elseif model.dims isa Union{Tuple, AbstractVector}
        collect(model.dims)
    else
        string(model.dims)
    end
    return Dict{String, Any}(
        "type" => "Dropout",
        "summary" => string(model),
        "p" => Float64(model.p),
        "dims" => dims_info,
    )
end

function LuxZarr.extract_model_info(model::Lux.NoOpLayer)
    return Dict{String, Any}(
        "type" => "NoOpLayer",
        "summary" => string(model),
    )
end

function LuxZarr.extract_model_info(nns::NamedTuple)
    d = Dict{String, Any}(
        "__is_namedtuple__" => true,
        "__keys__" => [string(k) for k in keys(nns)],
    )
    for (k, v) in pairs(nns)
        d[string(k)] = extract_model_info(v)
    end
    return d
end

_get_layer_list(layers::NamedTuple) = values(layers)
_get_layer_list(layers::Union{Tuple, AbstractVector}) = layers
_get_layer_list(layers) = (layers,)

function _normalize_identifier(s::AbstractString)
    str = strip(s)
    str = chopsuffix(chopprefix(str, "typeof("), ")")
    str = chopprefix(str, "Base.:")
    if contains(str, '.')
        str = split(str, '.')[end]
    end
    return str
end

const _ACTIVATION_MAP = Dict{String, Function}(
    "identity" => identity,
    "sigmoid" => Lux.NNlib.sigmoid_fast,
    "σ" => Lux.NNlib.sigmoid_fast,
    "silu" => Lux.NNlib.swish,
    "tanh" => Lux.NNlib.tanh_fast,
    "gelu_tanh" => Lux.NNlib.gelu,
    "gelu_accurate" => Lux.NNlib.gelu,
    [
        string(nameof(f)) => f for f in (
                Lux.NNlib.relu, Lux.NNlib.sigmoid_fast, Lux.NNlib.tanh_fast,
                Lux.NNlib.gelu, Lux.NNlib.leakyrelu, Lux.NNlib.swish,
                Lux.NNlib.softplus, Lux.NNlib.softsign, Lux.NNlib.celu,
                Lux.NNlib.elu, Lux.NNlib.mish, Lux.NNlib.selu,
                Lux.NNlib.lisht, Lux.NNlib.logsigmoid, Lux.NNlib.tanhshrink,
                Lux.NNlib.hardsigmoid, Lux.NNlib.hardswish,
            )
    ]...,
)

const _CONNECTION_MAP = Dict{String, Function}(
    "+" => +,
    "-" => -,
    "*" => *,
    "vcat" => vcat,
    "hcat" => hcat,
)

function _resolve_activation(act_str::AbstractString)
    key = _normalize_identifier(act_str)
    haskey(_ACTIVATION_MAP, key) && return _ACTIVATION_MAP[key]
    throw(ArgumentError("Cannot faithfully reconstruct model: unrecognized activation function '$(act_str)'."))
end

function _resolve_connection(conn_str::AbstractString)
    key = _normalize_identifier(conn_str)
    haskey(_CONNECTION_MAP, key) && return _CONNECTION_MAP[key]
    throw(ArgumentError("Cannot faithfully reconstruct model: unrecognized connection function '$(conn_str)'."))
end

# Layer reconstruction via dispatch
function LuxZarr.reconstruct_layer(::Val{:Dense}, info::AbstractDict; kwargs...)
    in_dims = Int(info["in_dims"])
    out_dims = Int(info["out_dims"])
    act = _resolve_activation(string(get(info, "activation", "identity")))
    use_bias = Bool(get(info, "use_bias", true))
    return Dense(in_dims => out_dims, act; use_bias = use_bias)
end

function LuxZarr.reconstruct_layer(::Val{:Conv}, info::AbstractDict; kwargs...)
    in_chs = Int(info["in_chs"])
    out_chs = Int(info["out_chs"])
    k_size = Tuple(Int(x) for x in info["kernel_size"])
    act = _resolve_activation(string(get(info, "activation", "identity")))
    use_bias = Bool(get(info, "use_bias", true))
    return Conv(k_size, in_chs => out_chs, act; use_bias = use_bias)
end

function _reconstruct_container(f, info::AbstractDict; kwargs...)
    haskey(info, "layers") || throw(ArgumentError("Container layer metadata missing 'layers' entry."))
    sub_layers = [reconstruct_model_from_info(l; kwargs...) for l in info["layers"]]
    return f(sub_layers)
end

LuxZarr.reconstruct_layer(::Val{:Chain}, info::AbstractDict; kwargs...) = _reconstruct_container(layers -> Chain(layers...), info; kwargs...)

function LuxZarr.reconstruct_layer(::Val{:Parallel}, info::AbstractDict; kwargs...)
    conn_str = string(get(info, "connection", "+"))
    op = _resolve_connection(conn_str)
    return _reconstruct_container(layers -> Parallel(op, layers...), info; kwargs...)
end

LuxZarr.reconstruct_layer(::Val{:BranchLayer}, info::AbstractDict; kwargs...) = _reconstruct_container(layers -> BranchLayer(layers...), info; kwargs...)

function LuxZarr.reconstruct_layer(::Val{:BatchNorm}, info::AbstractDict; kwargs...)
    chs = Int(get(info, "chs", 1))
    act = _resolve_activation(string(get(info, "activation", "identity")))
    ϵ = Float32(get(info, "epsilon", 1.0e-5))
    momentum = Float32(get(info, "momentum", 0.1))
    affine = Bool(get(info, "affine", true))
    track_stats = Bool(get(info, "track_stats", true))
    return BatchNorm(chs, act; epsilon = ϵ, momentum = momentum, affine = affine, track_stats = track_stats)
end

function LuxZarr.reconstruct_layer(::Val{:WrappedFunction}, info::AbstractDict; kwargs...)
    func_str = string(get(info, "func", "identity"))
    fn = _resolve_activation(func_str)
    return WrappedFunction(fn)
end

function LuxZarr.reconstruct_layer(::Val{:Dropout}, info::AbstractDict; kwargs...)
    p = Float32(get(info, "p", 0.5))
    dims_raw = get(info, "dims", ":")
    if dims_raw == ":" || isnothing(dims_raw) || isempty(dims_raw)
        return Dropout(p)
    elseif dims_raw isa Union{AbstractVector, Tuple}
        return Dropout(p; dims = Tuple(Int(d) for d in dims_raw))
    else
        return Dropout(p)
    end
end

function LuxZarr.reconstruct_layer(::Val{:NoOpLayer}, info::AbstractDict; kwargs...)
    return NoOpLayer()
end

function LuxZarr.reconstruct_layer(::Val{:SkipConnection}, info::AbstractDict; kwargs...)
    layer_info = get(info, "layers", get(info, "layer", nothing))
    isnothing(layer_info) && throw(ArgumentError("SkipConnection layer metadata missing sub-layer info."))
    sub_l = reconstruct_model_from_info(layer_info; kwargs...)
    conn_str = string(get(info, "connection", "+"))
    op = _resolve_connection(conn_str)
    return SkipConnection(sub_l, op)
end

function LuxZarr._setup_model_skeleton(model, root_g, ps, st, scalar_states, lazy::Bool = true)
    ps_skeleton, st_skeleton = if model isa NamedTuple
        nt_setups = map(m -> LuxCore.setup(default_rng(), m), model)
        (map(first, nt_setups), map(last, nt_setups))
    else
        LuxCore.setup(default_rng(), model)
    end
    attrs = Dict{String, Any}(root_g.attrs)
    scalar_params = get(attrs, "scalar_parameters", Dict{String, Any}())
    ps_loaded = isempty(ps) ? ps_skeleton : _guided_load_tree(root_g, ps_skeleton, "parameters", scalar_params, lazy)
    st_loaded = isempty(st) ? st_skeleton : _guided_load_tree(root_g, st_skeleton, "states", scalar_states, lazy)
    return (ps_loaded, st_loaded)
end

end # module
