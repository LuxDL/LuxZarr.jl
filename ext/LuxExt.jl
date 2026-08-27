module LuxExt

using LuxZarr: LuxZarr, extract_model_info, reconstruct_model_from_info, load_model,
               _keypath_to_path, _deserialize_scalar, _get_zarr_node, _isleaf
using Lux: Lux, AbstractLuxLayer, Dense, Conv, Chain, Parallel, BranchLayer, BatchNorm, SkipConnection
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
    rng=default_rng(),
    kwargs...,
)
    ps, st = Lux.setup(rng, model)
    return LuxZarr.load_model(store_or_path, ps, st; kwargs...)
end

function LuxZarr.extract_model_info(model::AbstractLuxLayer)
    info = Dict{String, Any}(
        "type" => string(nameof(typeof(model))),
        "summary" => string(model),
    )
    if model isa Lux.Dense
        info["in_dims"] = model.in_dims
        info["out_dims"] = model.out_dims
        info["use_bias"] = Lux.has_bias(model)
        info["activation"] = string(model.activation)
    elseif model isa Lux.Conv
        info["in_chs"] = model.in_chs
        info["out_chs"] = model.out_chs
        info["kernel_size"] = collect(model.kernel_size)
        info["activation"] = string(model.activation)
        info["use_bias"] = Lux.has_bias(model)
    elseif model isa Lux.SkipConnection
        info["layers"] = extract_model_info(model.layers)
        info["connection"] = string(model.connection)
    elseif model isa Lux.Chain || model isa Lux.Parallel || model isa Lux.BranchLayer || hasproperty(model, :layers)
        layers = model.layers
        layer_list = layers isa NamedTuple ? collect(values(layers)) : (layers isa Tuple || layers isa AbstractVector ? collect(layers) : [layers])
        info["layers"] = [extract_model_info(l) for l in layer_list]
    end
    return info
end

function _resolve_activation(act_str::AbstractString)
    if act_str == "identity" || act_str == "typeof(identity)"
        return identity
    elseif act_str == "relu" || act_str == "typeof(relu)"
        return Lux.NNlib.relu
    elseif act_str == "sigmoid" || act_str == "typeof(sigmoid)" || act_str == "sigmoid_fast"
        return Lux.NNlib.sigmoid_fast
    elseif act_str == "tanh" || act_str == "typeof(tanh)" || act_str == "tanh_fast"
        return Lux.NNlib.tanh_fast
    elseif act_str == "gelu" || act_str == "typeof(gelu)"
        return Lux.NNlib.gelu
    elseif act_str == "leakyrelu" || act_str == "typeof(leakyrelu)"
        return Lux.NNlib.leakyrelu
    elseif act_str == "swish" || act_str == "typeof(swish)" || act_str == "silu"
        return Lux.NNlib.silu
    else
        return identity
    end
end

function LuxZarr.reconstruct_model_from_info(info::AbstractDict)
    t = get(info, "type", "")
    if t == "Dense"
        in_dims = Int(info["in_dims"])
        out_dims = Int(info["out_dims"])
        act = _resolve_activation(string(get(info, "activation", "identity")))
        use_bias = Bool(get(info, "use_bias", true))
        return Dense(in_dims => out_dims, act; use_bias=use_bias)
    elseif t == "Conv"
        in_chs = Int(info["in_chs"])
        out_chs = Int(info["out_chs"])
        k_size = Tuple(Int(x) for x in info["kernel_size"])
        act = _resolve_activation(string(get(info, "activation", "identity")))
        use_bias = Bool(get(info, "use_bias", true))
        return Conv(k_size, in_chs => out_chs, act; use_bias=use_bias)
    elseif t == "Chain" && haskey(info, "layers")
        sub_layers = [reconstruct_model_from_info(l) for l in info["layers"]]
        if any(isnothing, sub_layers)
            return nothing
        end
        return Chain(sub_layers...)
    elseif t == "Parallel" && haskey(info, "layers")
        sub_layers = [reconstruct_model_from_info(l) for l in info["layers"]]
        if any(isnothing, sub_layers)
            return nothing
        end
        return Parallel(+, sub_layers...)
    elseif t == "BranchLayer" && haskey(info, "layers")
        sub_layers = [reconstruct_model_from_info(l) for l in info["layers"]]
        if any(isnothing, sub_layers)
            return nothing
        end
        return BranchLayer(sub_layers...)
    elseif t == "BatchNorm"
        chs = Int(get(info, "chs", 1))
        return BatchNorm(chs)
    elseif t == "SkipConnection" && (haskey(info, "layers") || haskey(info, "layer"))
        sub_l = reconstruct_model_from_info(get(info, "layers", get(info, "layer", nothing)))
        if sub_l === nothing
            return nothing
        end
        return SkipConnection(sub_l, +)
    end
    return nothing
end

function LuxZarr._setup_model_skeleton(model::AbstractLuxLayer, root_g, ps, st, scalar_states)
    _, st_skeleton = Lux.setup(default_rng(), model)
    if isempty(st)
        st = st_skeleton
    else
        st = Functors.fmap_with_path(st_skeleton; exclude=_isleaf) do kp, x
            rel_path = _keypath_to_path(kp)
            full_path = "states/" * rel_path
            z_node = _get_zarr_node(root_g, full_path)
            if z_node isa Zarr.ZArray
                arr_data = Array(z_node)
                return adapt(typeof(x), convert(AbstractArray{eltype(x)}, arr_data))
            elseif haskey(scalar_states, rel_path)
                return _deserialize_scalar(scalar_states[rel_path], typeof(x))
            end
            return x
        end
    end
    return (ps, st)
end

end # module
