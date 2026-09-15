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
Returns `nothing` if the model cannot be reconstructed or if `info` is `nothing`.
Can be extended via multiple dispatch for custom layers.
"""
function reconstruct_model_from_info(info::AbstractDict; kwargs...)
    if get(info, "__is_namedtuple__", false) == true
        raw_keys = get(info, "__keys__", nothing)
        keys_list = if raw_keys !== nothing
            [Symbol(k) for k in raw_keys]
        else
            [Symbol(k) for k in keys(info) if k != "__is_namedtuple__"]
        end
        vals_list = [reconstruct_model_from_info(info[string(k)]; kwargs...) for k in keys_list]
        return NamedTuple{Tuple(keys_list)}(vals_list)
    end
    t = Symbol(get(info, "type", ""))
    return _reconstruct_layer(Val(t), info; kwargs...)
end
reconstruct_model_from_info(::Nothing; kwargs...) = nothing
_reconstruct_layer(::Val, info::AbstractDict; kwargs...) = nothing

_get_model_summary(::Nothing) = nothing
function _get_model_summary(model)
    try
        return sprint(show, MIME("text/plain"), model)
    catch
        return string(model)
    end
end
