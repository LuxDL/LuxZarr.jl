_serialize_scalar(val::Symbol) = Dict("__type__" => "Symbol", "value" => string(val))
_serialize_scalar(::Val{B}) where {B} = Dict("__type__" => "Val", "value" => B)
_serialize_scalar(val::Union{Number, Bool, AbstractString}) = val
_serialize_scalar(val) = string(val)

_deserialize_typed_dict(::Val{:Symbol}, val::AbstractDict) = Symbol(val["value"])
_deserialize_typed_dict(::Val{:Val}, val::AbstractDict) = Val(val["value"])
_deserialize_typed_dict(type_str::AbstractString, val::AbstractDict) = _deserialize_typed_dict(Val(Symbol(type_str)), val)
_deserialize_typed_dict(::Val, val::AbstractDict) = val

_deserialize_scalar(val::AbstractDict, ::Type) = _deserialize_typed_dict(get(val, "__type__", ""), val)
_deserialize_scalar(val::AbstractString, ::Type{Symbol}) = startswith(val, ':') ? Symbol(SubString(val, 2)) : Symbol(val)
_deserialize_scalar(val::AbstractString, ::Type{Val}) = startswith(val, "Val{") ? (occursin("true", lowercase(val)) ? Val(true) : Val(false)) : Val(val)
_deserialize_scalar(val::AbstractString, ::Type{<:Val{B}}) where {B} = Val(B)
_deserialize_scalar(val::Number, ::Type{T}) where {T<:Number} = T(val)

function _deserialize_scalar(val::AbstractString, ::Type{Any})
    if startswith(val, ':')
        return Symbol(SubString(val, 2))
    elseif startswith(val, "Val{")
        return occursin("true", lowercase(val)) ? Val(true) : Val(false)
    end
    return val
end

_deserialize_scalar(val, ::Type) = val

