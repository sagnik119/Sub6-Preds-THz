function net = buildNet(options)
%=========================================================================%
% Build the neural network.
%=========================================================================%

switch options.type
    case 'MLP1'
        input = imageInputLayer(options.inputSize, 'Name', 'input');
        fc1 = fullyConnectedLayer(2048, 'Name', 'fc1');
        relu1 = reluLayer('Name','relu1');
        %relu1 = clippedReluLayer(10,"Name",'relu1');
        drop1 = dropoutLayer(0.4, 'Name', 'drop1');
        fc2 = fullyConnectedLayer(2048, 'Name', 'fc2');
        relu2 = reluLayer('Name','relu2');
        %relu2 = clippedReluLayer(10,"Name",'relu2');
        drop2 = dropoutLayer(0.4, 'Name', 'drop2');
        fc3 = fullyConnectedLayer(2048, 'Name', 'fc3');
        relu3 = reluLayer('Name','relu3');
        %relu3 = clippedReluLayer(10,"Name",'relu3');
        drop3 = dropoutLayer(0.4, 'Name', 'drop3');
        fc4 = fullyConnectedLayer(2048, 'Name', 'fc4');
        relu4 = reluLayer('Name','relu4');
        %relu4 = clippedReluLayer(10,"Name",'relu4');
        drop4 = dropoutLayer(0.4, 'Name', 'drop4');
        fc5 = fullyConnectedLayer(2048, 'Name', 'fc5');
        relu5 = reluLayer('Name','relu5');
        %relu5 = clippedReluLayer(10,"Name",'relu5');
        drop5 = dropoutLayer(0.4, 'Name', 'drop5');
%         fc6 = fullyConnectedLayer(options.numAnt(2), 'Name', 'fc6');
%         sfm = softmaxLayer('Name','sfm');
%         classifier = classificationLayer('Name','classifier');
        %fc6 = fullyConnectedLayer(options.outputSize(3), 'Name', 'fc6');
        fc6 = fullyConnectedLayer(options.outputSize(3), 'Name', 'fc6');
%         relu6 = reluLayer('Name','relu6');
        regressor = regressionLayer('Name', 'regressor');

        layers = [
                  input
                  fc1
                  relu1
                  drop1
                  fc2
                  relu2
                  drop2
                  fc3
                  relu3
                  drop3
                  fc4
                  relu4
                  drop4
                  fc5
                  relu5
                  drop5
                  fc6
                  %relu6
                  regressor
                 ];
        net = layerGraph(layers);

end