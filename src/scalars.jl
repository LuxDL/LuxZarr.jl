_serialize_scalar(val::Symbol) = Dict("__type__" => "Symbol", "value" => string(val))
_serialize_scalar(::Val{B}) where {B} = Dict("__type__" => "Val", "value" => B)
_serialize_scalar(val::Union{Number, Bool, AbstractString}) = val
_serialize_scalar(val) = string(val)

function _deserialize_scalar(val, ::Type{T}) where {T}
    if val isa Dict
        type_str = get(val, "__type__", "")
        if type_str == "Symbol"
            return Symbol(val["value"])
        elseif type_str == "Val"
            return Val(val["value"])
        end
    end
    if (T <: Symbol || T == Any) && val isa AbstractString && startswith(val, ":")
        return Symbol(SubString(val, 2))
    end
    if T <: Symbol && val isa AbstractString
        return Symbol(val)
    end
    if (T <: Val || T == Any) && val isa AbstractString && startswith(val, "Val{")
        return occursin("true", lowercase(val)) ? Val(true) : Val(false)
    end
    if T !== Any && val isa Number
        return T(val)
    end
    return val
end
