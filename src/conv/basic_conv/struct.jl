
mutable struct BConv

    kernel::Array{Float32, 3}

    kernel_grad::Union{Array{Float32, 3}, Nothing}

    input::Union{Array{Float32, 3}, Nothing}
    z::Union{Array{Float32, 3}, Nothing}
    a::Union{Array{Float32, 3}, Nothing}

    function BConv(kernel::Array{Float32, 3})
        return new(kernel, nothing, nothing, nothing, nothing)
    end
end

function BConv(kernel_y::Int, kernel_x::Int)
    kernel = randn(Float32, 1, kernel_y, kernel_x) * 0.5f0
    return BConv(kernel)
end

