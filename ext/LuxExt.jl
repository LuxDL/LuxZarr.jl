module LuxExt

using LuxZarr: LuxZarr, extract_model_info, reconstruct_model_from_info, load_model,
               _keypath_to_path, _deserialize_scalar, _get_zarr_node, _isleaf, _guided_load_tree,
               LazyParameters, LazyState, LazyLuxModel, unwrap
using Lux: Lux, AbstractLuxLayer, Dense, Conv, Chain, Parallel, BranchLayer, BatchNorm, SkipConnection
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
    model::AbstractLuxLayer;
    lazy::Bool=true,
    rng=default_rng(),
    kwargs...,
)
    ps, st = Lux.setup(rng, model)
    ps_loaded, st_loaded = LuxZarr.load_model(store_or_path, ps, st; lazy=lazy, kwargs...)
    if lazy
        return LazyLuxModel(model, ps_loaded, st_loaded)
    else
        return (unwrap(ps_loaded), unwrap(st_loaded))
    end
end

# Callable LazyLuxModel
(lm::LazyLuxModel)(x, ps=lm.ps, st=lm.st) = lm.model(x, Base.materialize(ps), unwrap(st))

# Transparent application of AbstractLuxLayer with LazyParameters / LazyState
(l::LuxCore.AbstractLuxLayer)(x, ps::LazyParameters, st::LazyState) = l(x, Base.materialize(ps), unwrap(st))
(l::LuxCore.AbstractLuxLayer)(x, ps::LazyParameters, st=NamedTuple()) = l(x, Base.materialize(ps), unwrap(st))
(l::LuxCore.AbstractLuxLayer)(x, ps, st::LazyState) = l(x, Base.materialize(ps), unwrap(st))

(l::LuxCore.AbstractLuxWrapperLayer)(x, ps::LazyParameters, st::LazyState) = l(x, Base.materialize(ps), unwrap(st))
(l::LuxCore.AbstractLuxWrapperLayer)(x, ps::LazyParameters, st=NamedTuple()) = l(x, Base.materialize(ps), unwrap(st))
(l::LuxCore.AbstractLuxWrapperLayer)(x, ps, st::LazyState) = l(x, Base.materialize(ps), unwrap(st))

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

_get_layer_list(layers::NamedTuple) = values(layers)
_get_layer_list(layers::Union{Tuple, AbstractVector}) = layers
_get_layer_list(layers) = (layers,)

const _ACTIVATION_MAP = Dict{String, Function}(
    "identity" => identity,
    "typeof(identity)" => identity,
    "relu" => Lux.NNlib.relu,
    "typeof(relu)" => Lux.NNlib.relu,
    "NNlib.relu" => Lux.NNlib.relu,
    "sigmoid" => Lux.NNlib.sigmoid_fast,
    "typeof(sigmoid)" => Lux.NNlib.sigmoid_fast,
    "sigmoid_fast" => Lux.NNlib.sigmoid_fast,
    "typeof(sigmoid_fast)" => Lux.NNlib.sigmoid_fast,
    "NNlib.sigmoid_fast" => Lux.NNlib.sigmoid_fast,
    "σ" => Lux.NNlib.sigmoid_fast,
    "typeof(σ)" => Lux.NNlib.sigmoid_fast,
    "NNlib.σ" => Lux.NNlib.sigmoid_fast,
    "tanh" => Lux.NNlib.tanh_fast,
    "typeof(tanh)" => Lux.NNlib.tanh_fast,
    "tanh_fast" => Lux.NNlib.tanh_fast,
    "typeof(tanh_fast)" => Lux.NNlib.tanh_fast,
    "NNlib.tanh_fast" => Lux.NNlib.tanh_fast,
    "gelu" => Lux.NNlib.gelu,
    "typeof(gelu)" => Lux.NNlib.gelu,
    "NNlib.gelu" => Lux.NNlib.gelu,
    "gelu_tanh" => Lux.NNlib.gelu,
    "typeof(gelu_tanh)" => Lux.NNlib.gelu,
    "NNlib.gelu_tanh" => Lux.NNlib.gelu,
    "gelu_accurate" => Lux.NNlib.gelu,
    "typeof(gelu_accurate)" => Lux.NNlib.gelu,
    "NNlib.gelu_accurate" => Lux.NNlib.gelu,
    "leakyrelu" => Lux.NNlib.leakyrelu,
    "typeof(leakyrelu)" => Lux.NNlib.leakyrelu,
    "NNlib.leakyrelu" => Lux.NNlib.leakyrelu,
    "swish" => Lux.NNlib.swish,
    "typeof(swish)" => Lux.NNlib.swish,
    "NNlib.swish" => Lux.NNlib.swish,
    "silu" => Lux.NNlib.swish,
    "typeof(silu)" => Lux.NNlib.swish,
)

_resolve_activation(act_str::AbstractString) = get(_ACTIVATION_MAP, act_str, identity)

# Layer reconstruction via dispatch
function LuxZarr.reconstruct_model_from_info(info::AbstractDict)
    t = Symbol(get(info, "type", ""))
    return _reconstruct_layer(Val(t), info)
end

_reconstruct_layer(::Val, info::AbstractDict) = nothing

function _reconstruct_layer(::Val{:Dense}, info::AbstractDict)
    in_dims = Int(info["in_dims"])
    out_dims = Int(info["out_dims"])
    act = _resolve_activation(string(get(info, "activation", "identity")))
    use_bias = Bool(get(info, "use_bias", true))
    return Dense(in_dims => out_dims, act; use_bias=use_bias)
end

function _reconstruct_layer(::Val{:Conv}, info::AbstractDict)
    in_chs = Int(info["in_chs"])
    out_chs = Int(info["out_chs"])
    k_size = Tuple(Int(x) for x in info["kernel_size"])
    act = _resolve_activation(string(get(info, "activation", "identity")))
    use_bias = Bool(get(info, "use_bias", true))
    return Conv(k_size, in_chs => out_chs, act; use_bias=use_bias)
end

function _reconstruct_container(f, info::AbstractDict)
    haskey(info, "layers") || return nothing
    sub_layers = [reconstruct_model_from_info(l) for l in info["layers"]]
    any(isnothing, sub_layers) && return nothing
    return f(sub_layers)
end

_reconstruct_layer(::Val{:Chain}, info::AbstractDict) = _reconstruct_container(layers -> Chain(layers...), info)
_reconstruct_layer(::Val{:Parallel}, info::AbstractDict) = _reconstruct_container(layers -> Parallel(+, layers...), info)
_reconstruct_layer(::Val{:BranchLayer}, info::AbstractDict) = _reconstruct_container(layers -> BranchLayer(layers...), info)

function _reconstruct_layer(::Val{:BatchNorm}, info::AbstractDict)
    chs = Int(get(info, "chs", 1))
    return BatchNorm(chs)
end

function _reconstruct_layer(::Val{:SkipConnection}, info::AbstractDict)
    layer_info = get(info, "layers", get(info, "layer", nothing))
    layer_info === nothing && return nothing
    sub_l = reconstruct_model_from_info(layer_info)
    sub_l === nothing && return nothing
    return SkipConnection(sub_l, +)
end

function LuxZarr._setup_model_skeleton(model::AbstractLuxLayer, root_g, ps, st, scalar_states, lazy::Bool=true)
    _, st_skeleton = Lux.setup(default_rng(), model)
    st = isempty(st) ? st_skeleton : _guided_load_tree(root_g, st_skeleton, "states", scalar_states, lazy)
    return (ps, st)
end

end # module
