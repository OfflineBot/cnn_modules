
mutable struct BConvBias

    kernel::Array{Float32, 3}
    bias::Vector{Float32}

    kernel_grad::Union{Array{Float32, 3}, Nothing}

    input::Union{Array{Float32, 3}, Nothing}
    z::Union{Array{Float32, 3}, Nothing}
    a::Union{Array{Float32, 3}, Nothing}

    function BConvBias(kernel::Array{Float32, 3}, bias::Vector{Float32})
        return new(kernel, bias, nothing, nothing, nothing, nothing)
    end
end

function BConvBias(kernel_y::Int, kernel_x::Int)
    kernel = randn(Float32, 1, kernel_y, kernel_x) * 0.5f0
    bias = randn(Float32, 1)
    return BConvBias(kernel, bias)
end

