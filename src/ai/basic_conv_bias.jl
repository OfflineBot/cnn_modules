
function basic_conv_bias()

    lr = 0.01f0

    data = RImage(10, (20, 20), 10, 0.5f0) # 10, 20, 20

    layer1 = BConvBias(5, 5)
    layer2 = BConvBias(5, 5)
    layer3 = DenseLayer(12*12, 10)

    for i in 1:100_000

        a1 = bforward!(layer1, data.input) # -> 10, 16, 16
        a2 = bforward!(layer2, a1) # -> 10, 12, 12
        a2 = reshape_output(a2, (10, 12*12))
        a3 = forward!(layer3, a2, relu)

        loss = mse_loss(a3, data.output)
        if i % 1000 == 0
            perc = Int(round(i / 100_000 * 100))
            println("I: $perc% | Loss: $loss")
        end

        delta3 = mse_backward(layer3, a3, data.output)
        delta3 = backward!(layer3, delta3, deriv_relu)
        delta3 = reshape_delta(layer2, delta3)
        delta2 = bbackward!(layer2, delta3) # <- 10, 12, 12
        bbackward!(layer1, delta2) # <- 10, 16, 16
        update!(layer3, lr)
        bupdate!(layer2, lr)
        bupdate!(layer1, lr)
    end

end

