
export fully_connected

function fully_connected()
    xor = XOR()
    lr = 0.01f0

    layer1 = DenseLayer(2, 10)
    layer2 = DenseLayer(10, 1)

    for _ in 1:1000
        a1 = forward!(layer1, no_zero_input(xor, 0.1f0), relu)
        a2 = forward!(layer2, a1, identity)

        loss = mse_loss(a2, no_zero_output(xor, 0.1f0))

        if loss < 0.1 
            println(loss)
            println(a2)
        end

        delta2 = mse_backward(layer2, a2, no_zero_output(xor, 0.1f0))
        delta2 = backward!(layer2, delta2, deriv_relu)
        backward!(layer1, delta2, identity)

        update!(layer1, lr)
        update!(layer2, lr)
    end
end
