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
    reconstruct_model_from_info(info) -> Union{Any, Nothing}

Reconstruct a model layer struct from metadata dictionary `info`.
Returns `nothing` if the model cannot be reconstructed or if `info` is `nothing`.
Can be extended via multiple dispatch for custom layers.
"""
reconstruct_model_from_info(info) = nothing

_get_model_summary(::Nothing) = nothing
function _get_model_summary(model)
    try
        return sprint(show, MIME("text/plain"), model)
    catch
        return string(model)
    end
end
