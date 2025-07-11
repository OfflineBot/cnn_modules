
export DenseLayer

mutable struct DenseLayer
    weight::Matrix{Float32}
    bias::Matrix{Float32}

    input_data::Union{Matrix{Float32}, Nothing}
    z::Union{Matrix{Float32}, Nothing}
    a::Union{Matrix{Float32}, Nothing}
    delta::Union{Matrix{Float32}, Nothing}
    grad_weight::Union{Matrix{Float32}, Nothing}
    grad_bias::Union{Matrix{Float32}, Nothing}

    function DenseLayer(weight::Matrix{Float32}, bias::Matrix{Float32})
        return new(weight, bias, nothing, nothing, nothing, nothing, nothing, nothing)
    end

end

function DenseLayer(input_size::Int, output_size::Int)
    weight::Matrix{Float32} = randn(input_size, output_size) .* 0.5f0
    bias::Matrix{Float32} = randn(1, output_size) .* 0.5f0
    return DenseLayer(weight, bias)
end

