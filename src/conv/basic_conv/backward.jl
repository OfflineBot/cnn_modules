
export bbackward!, reshape_delta


function reshape_delta(layer::BConv, delta::Matrix{Float32})::Array{Float32, 3}

    layer_shape = size(layer.a)
    layer_length = length(layer.a)
    delta_length = length(delta)

    if (layer_length != delta_length)
        @show size(layer.a)
        @show size(delta)
        error("Shapes dont match for reshaping data")
    end

    return reshape(delta, layer_shape)

end


function bbackward!(layer::BConv, delta::Array{Float32, 3})::Array{Float32, 3}
    kernel = layer.kernel

    delta .*= deriv_relu(layer.z)

    flipped_kernel = reverse(reverse(kernel[1, :, :], dims=1), dims=2)

    kernel_shape = size(flipped_kernel)
    input_shape = size(delta)
    output_shape_y = input_shape[2] + kernel_shape[1] - 1
    output_shape_x = input_shape[3] + kernel_shape[2] - 1

    new_delta = zeros(Float32, input_shape[1], output_shape_y, output_shape_x)

    for i in 1:size(delta)[1]
        conv_t = conv_transpose(delta[i, :, :], flipped_kernel)
        new_delta[i, :, :] = conv_t
    end

    layer.kernel_grad = cross_correlation(layer.input, new_delta)
    return new_delta
end


function conv_transpose(delta::Matrix{Float32}, kernel::Matrix{Float32})::Matrix{Float32}

    delta_shape = size(delta)
    kernel_shape = size(kernel)
    out_y = delta_shape[1] + kernel_shape[1] - 1
    out_x = delta_shape[2] + kernel_shape[2] - 1

    out_mat = zeros(Float32, out_y, out_x)

    for i in 1:delta_shape[1]
        for j in 1:delta_shape[2]
            out_mat[i:i+kernel_shape[1]-1, j:j+kernel_shape[2] - 1] .+= delta[i, j] * kernel
        end
    end

    return out_mat
end


function cross_correlation(input::Array{Float32, 3}, delta::Array{Float32, 3})::Array{Float32, 3}
    input_batch_count, input_height, input_width = size(input)
    delta_batch_count, delta_height, delta_width = size(delta)
    @assert input_batch_count == delta_batch_count "Batch sizes must match for Cross Correlation"

    kernel_height = input_height - delta_height + 1
    kernel_width = input_width - delta_width + 1

    output = zeros(Float32, 1, kernel_height, kernel_width)

    for image in 1:input_batch_count
        input_2d_slice = @view input[image, :, :]
        delta_2d_slice = @view delta[image, :, :]
        for i in 1:kernel_height
            for j in 1:kernel_width
                input_patch = @view input_2d_slice[
                    i:i+delta_height - 1, 
                    j:j+delta_width - 1]
                output[1, i, j] += sum(input_patch .* delta_2d_slice)
            end
        end
    end

    return output
end

