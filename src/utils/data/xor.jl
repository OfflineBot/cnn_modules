
export XOR
export input_shape, output_shape

mutable struct XOR

    input::Matrix{Float32}
    output::Matrix{Float32}

    function XOR()
        input = [
            1 1;
            0 0;
            1 0;
            0 1;
        ]
        output_raw = [
            0; 0; 1; 1;
        ]
        output = reshape(output_raw, :, 1)
        return new(input, output)

    end

end


input_shape(data::XOR)::Vector{Int} = 
    [size(data.input)[1], size(data.input)[2]]


output_shape(data::XOR)::Vector{Int} = 
    [size(data.output)[1], size(data.output)[2]]


no_zero_input(data::XOR, no_zero::Float32)::Matrix{Float32} =
    ifelse.(data.input .== 0.f0, no_zero, data.input)


no_zero_output(data::XOR, no_zero::Float32)::Matrix{Float32} =
    reshape(ifelse.(data.output .== 0.f0, no_zero, data.output), :, 1)

