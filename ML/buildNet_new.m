% CNN model 

function net = buildnet (options)
    % Returns CNN model
    % Defining individual layers
    input = imageInputLayer(options.inputSize, "Name", "input");
    
    conv1 = convolution2dLayer([1, 5], 16, "Name", "conv1");
    norm1 = batchNormalizationLayer("Name", "norm1");
    relu1 = reluLayer("Name", "relu1");
    maxpool1 = maxPooling2dLayer([1, 2], "Stride", [1, 2], "Name", "maxpool1");
    
    conv2 = convolution2dLayer([1, 5], 32, "Name", "conv2");
    norm2 = batchNormalizationLayer("Name", "norm2");
    relu2 = reluLayer("Name", "relu2");
    maxpool2 = maxPooling2dLayer([1, 2], "Stride", [1, 2], "Name", "maxpool2");
    
    conv3 = convolution2dLayer([1, 5], 64, "Name", "conv3");
    norm3 = batchNormalizationLayer("Name", "norm3");
    relu3 = reluLayer("Name", "relu3");
    maxpool3 = maxPooling2dLayer([2, 2], "Stride", [1, 2], "Name", "maxpool3");
    
    conv4 = convolution2dLayer([2, 2], 64, "Name", "conv4");
    norm4 = batchNormalizationLayer("Name", "norm4");
    relu4 = reluLayer("Name", "relu4");
    
    conv5 = convolution2dLayer([3, 3], 128, "Name", "conv5");
    norm5 = batchNormalizationLayer("Name", "norm5");
    relu5 = reluLayer("Name", "relu5");
    maxpool5 = maxPooling2dLayer([2, 2], "Stride", [2, 2], "Name", "maxpool5");
    
    conv6 = convolution2dLayer([3, 3], 256, "Name", "conv6");
    norm6 = batchNormalizationLayer("Name", "norm6");
    relu6 = reluLayer("Name", "relu6");
    maxpool6 = maxPooling2dLayer([2, 2], "Stride", [2, 2], "Name", "maxpool6");
    
    conv7 = convolution2dLayer([3, 3], 512, "Name", "conv7");
    norm7 = batchNormalizationLayer("Name", "norm7");
    relu7 = reluLayer("Name", "relu7");
    maxpool7 = maxPooling2dLayer([2, 2], "Stride", [2, 2], "Name", "maxpool7");
    
    conv8 = convolution2dLayer([3, 3], 512, "Name", "conv8");
    norm8 = batchNormalizationLayer("Name", "norm8");
    relu8 = reluLayer("Name", "relu8");
    maxpool8 = maxPooling2dLayer([2, 2], "Stride", [2, 2], "Name", "maxpool8");
    
    fc1 = fullyConnectedLayer(options.outputSize(3), "Name", "fc1");
    regressor = regressionLayer("Name", "regressor");
    
    layers = [
        input
        conv1
        norm1
        relu1
        maxpool1
        conv2
        norm2
        relu2
        maxpool2
        conv3
        norm3
        relu3
        maxpool3
        conv4
        norm4
        relu4
        conv5
        norm5
        relu5
        maxpool5
        conv6
        norm6
        relu6
        maxpool6
        conv7
        norm7
        relu7
        maxpool7
        conv8
        norm8
        relu8
        maxpool8
        fc1
        regressor
    ];
    
    net = layerGraph(layers);

end
