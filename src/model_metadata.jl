"""
    extract_model_info(model) -> Union{Dict{String, Any}, Nothing}

Extract architectural metadata from a model layer for saving into Zarr attributes.
Returns `nothing` if `model` is `nothing`. For generic models, returns a dictionary containing
the model's type name and string representation. Can be extended via multiple dispatch for custom layers.
"""
function extract_model_info(model)
    return Dict{String, Any}(
        "type" => string(nameof(typeof(model))),
        "summary" => string(model),
    )
end
extract_model_info(::Nothing) = nothing

"""
    reconstruct_model_from_info(info; kwargs...) -> Union{Any, Nothing}

Reconstruct a model layer struct from metadata dictionary `info`.
Returns `nothing` if `info` is `nothing`. Throws an `ArgumentError` if a layer cannot be faithfully reconstructed.
Can be extended via multiple dispatch by defining `LuxZarr.reconstruct_layer(::Val{:LayerType}, info)`.
"""
function reconstruct_model_from_info(info::AbstractDict; kwargs...)
    if get(info, "__is_namedtuple__", false) == true
        raw_keys = get(info, "__keys__", nothing)
        keys_list = if !isnothing(raw_keys)
            [Symbol(k) for k in raw_keys]
        else
            [Symbol(k) for k in keys(info) if k != "__is_namedtuple__"]
        end
        vals_list = [reconstruct_model_from_info(info[string(k)]; kwargs...) for k in keys_list]
        return NamedTuple{Tuple(keys_list)}(vals_list)
    end
    t_str = get(info, "type", "")
    if isempty(t_str)
        throw(ArgumentError("Cannot reconstruct model: missing 'type' in layer metadata info."))
    end
    t = Symbol(t_str)
    return reconstruct_layer(Val(t), info; kwargs...)
end
reconstruct_model_from_info(::Nothing; kwargs...) = nothing

function reconstruct_layer(::Val{T}, info::AbstractDict; kwargs...) where {T}
    throw(ArgumentError("Cannot faithfully reconstruct layer of type '$(T)'. Define `LuxZarr.reconstruct_layer(::Val{:$T}, info)` or load into a pre-constructed model via `load_model(path, model)`."))
end

"""
    resolve_activation(act_str::AbstractString) -> Function

Resolve a string representation of an activation function (e.g., `"relu"`, `"tanh"`, `"identity"`)
to its corresponding function. Throws an `ArgumentError` if the activation is unrecognized or unsupported.
Requires `Lux` extension to be loaded for standard neural network activations, or custom dispatch definitions.
"""
function resolve_activation end

"""
    resolve_connection(conn_str::AbstractString) -> Function

Resolve a string representation of a container connection function (e.g., `"+"`, `"*"`, `"vcat"`)
to its corresponding function. Throws an `ArgumentError` if the connection is unrecognized or unsupported.
Requires `Lux` extension to be loaded for standard container connections, or custom dispatch definitions.
"""
function resolve_connection end

_get_model_summary(::Nothing) = nothing
function _get_model_summary(model)
    try
        return sprint(show, MIME("text/plain"), model)
    catch
        return string(model)
    end
end
