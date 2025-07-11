
export bforward!

function reshape_output(output::Array{Float32, 3}, target::Tuple{Int, Int})
    output_len = length(output)
    target_len = target[1] * target[2]
    @assert output_len == target_len "Dimensions dont match for reshape"

    return reshape(output, target)
end

function bforward!(layer::BConv, input::Array{Float32, 3})::Array{Float32, 3}
    layer.input = input

    input_shape = size(input)
    kernel_shape = size(layer.kernel)
    output_shape_y = input_shape[2] - kernel_shape[2] + 1
    output_shape_x = input_shape[3] - kernel_shape[3] + 1

    kernel = layer.kernel[1, :, :]

    z = zeros(Float32, input_shape[1], output_shape_y, output_shape_x)

    for i in 1:input_shape[1]
        input_matrix = layer.input[i, :, :]
        output = bconvolute(kernel, input_matrix)
        z[i, :, :] = output
    end

    layer.z = z
    a = relu(z)
    layer.a = a

    return a
end


function bconvolute(kernel::Matrix{Float32}, input::Matrix{Float32})::Matrix{Float32}
    shape = size(input)
    kernel_shape = size(kernel)
    out_y = shape[1] - kernel_shape[1] + 1
    out_x = shape[2] - kernel_shape[2] + 1

    out_mat = zeros(Float32, out_y, out_x)

    for i in 1:out_y
        for j in 1:out_x
            input_patch = @view input[i:i+kernel_shape[1]-1, j:j+kernel_shape[2]-1]
            out_mat[i, j] = sum(input_patch .* kernel)
        end
    end

    return out_mat
end

