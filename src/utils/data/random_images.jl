

export RImage
export input_shape, output_shape

mutable struct RImage

    input::Array{Float32, 3}
    output::Matrix{Float32}

    function RImage(items::Int, in_shape::Tuple{Int, Int}, out_shape::Int, modifier::Float32=1.f0)
        input = randn(items, in_shape[1], in_shape[2]) * modifier
        output = randn(items, out_shape) * modifier
        return new(input, output)

    end

end


input_shape(data::RImage)::Vector{Int} = 
    [size(data.input)[1], size(data.input)[2], size(data.input)[3]]


output_shape(data::RImage)::Vector{Int} = 
    [size(data.output)[1], size(data.output)[2], size(data.output)[3]]


no_zero_input(data::RImage, no_zero::Float32)::Matrix{Float32} =
    ifelse.(data.input .== 0.f0, no_zero, data.input)


no_zero_output(data::RImage, no_zero::Float32)::Matrix{Float32} =
    reshape(ifelse.(data.output .== 0.f0, no_zero, data.output), :, 1)

