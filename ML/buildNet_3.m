% Dense NN model to determine best beamformer from NORMALIZED THz channel factors
function net = buildnet_3 (options)
    % Returns NN model
    % Defining individual layers
    input = featureInputLayer(options.num_THz_factors_used,"Name",'input'); % TODO : may eliminate phase and see if performance remains same or not

    fc1 = fullyConnectedLayer(64, 'Name','fc1');
    relu1 = reluLayer('Name','relu1');

    fc2 = fullyConnectedLayer(128, 'Name','fc2');
    relu2 = reluLayer('Name','relu2');

    fc3 = fullyConnectedLayer(256, 'Name','fc3');
    relu3 = reluLayer('Name','relu3');

    fc4 = fullyConnectedLayer(options.num_beamformers, 'Name', 'fc4');
    softmax1 = softmaxLayer('Name','softmax1');
    classifier = classificationLayer('Name', 'classifier');

    layers = [
                input
                fc1
                relu1
                fc2
                relu2
                fc3
                relu3
                fc4
                softmax1
                classifier
        ];
    net = layerGraph(layers);
end
