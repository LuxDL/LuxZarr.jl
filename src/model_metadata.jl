"""
    extract_model_info(model) -> Optional{Dict{String, Any}}

Extract architectural metadata from a model layer for saving into Zarr attributes.
Returns `nothing` if the model type does not have a dedicated extractor.
Can be extended for custom user layers.
"""
function extract_model_info(model)
    return Dict{String, Any}(
        "type" => string(nameof(typeof(model))),
        "summary" => string(model),
    )
end
extract_model_info(::Nothing) = nothing

"""
    reconstruct_model_from_info(info) -> Optional{Any}

Reconstruct a model layer struct from metadata dictionary `info`.
Returns `nothing` if the model cannot be reconstructed.
Can be extended for custom user layers.
"""
function reconstruct_model_from_info(info)
    return nothing
end

function _get_model_summary(model)
    if model === nothing
        return nothing
    end
    try
        return sprint(show, MIME("text/plain"), model)
    catch
        return string(model)
    end
end
