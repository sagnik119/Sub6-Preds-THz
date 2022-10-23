% CNN model 

function net = buildnet_2 (options)
    % Returns CNN model
    % Defining individual layers
    input = imageInputLayer(options.inputSize,"Name",'input');

    conv1 = convolution2dLayer([1,5],16, 'Name','conv1');
    norm1 = batchNormalizationLayer('Name', 'norm1');
    relu1 = reluLayer('Name','relu1');
    maxpool1 = maxPooling2dLayer([1,2],'Stride',[1,2], 'Name','maxpool1');

    conv2 = convolution2dLayer([1,5],32, 'Name','conv2');
    norm2 = batchNormalizationLayer('Name', 'norm2');
    relu2 = reluLayer('Name','relu2');
    maxpool2 = maxPooling2dLayer([1,2],'Stride',[1,2],'Name','maxpool2');

    conv3 = convolution2dLayer([1,5],64, 'Name','conv3');
    norm3 = batchNormalizationLayer('Name', 'norm3');
    relu3 = reluLayer('Name','relu3');
    maxpool3 = maxPooling2dLayer([2,2],'Stride',[1,2],'Name','maxpool3');

    conv4 = convolution2dLayer([2,2],64, 'Name','conv4');
    norm4 = batchNormalizationLayer('Name', 'norm4');
    relu4 = reluLayer('Name','relu4');

    fc1 = fullyConnectedLayer(options.outputSize(3), 'Name', 'fc1');
    relu5 = reluLayer('Name','relu5');
    regressor = regressionLayer('Name', 'regressor');

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
                fc1
                relu5
                regressor
        ];
    net = layerGraph(layers);
end
