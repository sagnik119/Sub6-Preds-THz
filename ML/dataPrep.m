function [dataset,options] = dataPrep(fileName,options)
%=========================================================================%
% dataPrep is data preparing function.
% INPUTS:
%	fileName: a single-field data struct with data
%		  file paths.
%	options: a MATLAB structure with the experiment settings
% OUTPUTS:
%	dataset: dataset structure for training and validataion data
%	options: updated options structure
%=========================================================================%
len = length(fileName);
d = {};
%loc = {};
labels = {};
DoD_phi = {};
DoD_theta = {};
DoA_phi = {};
DoA_theta = {};
phase = {};
ToA = {};
power = {};
for i = 1:len% Normalize data
	load(fileName(i).name)
	x = rawData.channel;
    %loc(i) = {rawData.userLoc};
    if strcmp( options.case, 'NLOS' ) 
        labels(i) = {rawData.labels};
    end
    DoD_phi(i) = {rawData.DoD_phi};
    DoD_theta(i) = {rawData.DoD_theta};
    DoA_phi(i) = {rawData.DoA_phi};
    DoA_theta(i) = {rawData.DoA_theta};
    phase(i) = {rawData.phase};
    ToA(i) = {rawData.ToA};
    power(i) = {rawData.power};
	d(i) = {x};
end

dataset.data = d;
%dataset.userLoc = loc;
dataset.labels = labels;
dataset.DoD_phi = DoD_phi;
dataset.DoD_theta = DoD_theta;
dataset.DoA_phi = DoA_phi;
dataset.DoA_theta = DoA_theta;
dataset.phase = phase;
dataset.ToA = ToA;
dataset.power = power;
clear d loc labels

% Shuffling data:
% ---------------
options.numSamples = size( dataset.data{1},3 );
% Check if previous shuffling information is already stored, else create
% and store for future
% if ~exist("shuffledInd.mat","file")
%     shuffledInd = randperm(options.numSamples);
%     save ("shuffledInd.mat", "shuffledInd", '-mat');
% else
%     load("shuffledInd.mat", "shuffledInd");
% end
% options.shuffledInd = shuffledInd;
shuffledInd = randperm(options.numSamples);
disp(shuffledInd(1));
for i = 1:len
    dataset.data{i} = dataset.data{i}(:,:,shuffledInd);
    %dataset.labels{i} = dataset.labels{i}(shuffledInd);
    dataset.DoD_phi{i} = dataset.DoD_phi{i}(shuffledInd);
    dataset.DoD_theta{i} = dataset.DoD_theta{i}(shuffledInd);
    dataset.DoA_phi{i} = dataset.DoA_phi{i}(shuffledInd);
    dataset.DoA_theta{i} = dataset.DoA_theta{i}(shuffledInd);
    dataset.phase{i} = dataset.phase{i}(shuffledInd);
    dataset.ToA{i} = dataset.ToA{i}(shuffledInd);
    dataset.power{i} = dataset.power{i}(shuffledInd);
    %TODO Include LOS shuffle info for blockage prediction?
    %dataset.userLoc{i} = dataset.userLoc{i}(:,shuffledInd);
end

% Divide data:
% ------------
numTrain = floor( (1 - options.valPer)*options.numSamples );
options.numOfTrain = numTrain;
options.numOfVal = options.numSamples - options.numOfTrain;
sub6Train = dataset.data{1}(:,:,1:numTrain);% Sub-6 training channels
sub6Val = dataset.data{1}(:,:,numTrain+1:end);% Sub-6 validation channels

% sub6TrainLoc = dataset.userLoc{1}(:,1:numTrain);% Sub-6 training user locations
% sub6ValLoc = dataset.userLoc{1}(:,numTrain+1:end);% Sub-6 validation user locations
% dataset.trainInpLoc = sub6TrainLoc;
% dataset.valInpLoc = sub6ValLoc;
if len > 1
    highTrain = dataset.data{2}(:,:,1:numTrain);% High training channels
    highVal = dataset.data{2}(:,:,numTrain+1:end);% High validation channels
    highTrainDoD_phi = dataset.DoD_phi{2}(1:numTrain);
    highValDoD_phi = dataset.DoD_phi{2}(numTrain+1:end);
    highTrainDoD_theta = dataset.DoD_theta{2}(1:numTrain);
    highValDoD_theta = dataset.DoD_theta{2}(numTrain+1:end);
    highTrainDoA_phi = dataset.DoA_phi{2}(1:numTrain);
    highValDoA_phi = dataset.DoA_phi{2}(numTrain+1:end);
    highTrainDoA_theta = dataset.DoA_theta{2}(1:numTrain);
    highValDoA_theta = dataset.DoA_theta{2}(numTrain+1:end);
    highTrainPhase = dataset.phase{2}(1:numTrain);
    highValPhase = dataset.phase{2}(numTrain+1:end);
    highTrainToA = dataset.ToA{2}(1:numTrain);
    highValToA = dataset.ToA{2}(numTrain+1:end);
    highTrainPower = dataset.power{2}(1:numTrain);
    highValPower = dataset.power{2}(numTrain+1:end);
%     highTrainLoc = dataset.userLoc{2}(:,1:numTrain);% High training user locations
%     highValLoc = dataset.userLoc{2}(:,numTrain+1:end);% High validation user locations
%     dataset.trainOutLoc = highTrainLoc;
%     dataset.valOutLoc = highValLoc;
end

% Compute data statistics:
% ------------------------
abs_value = abs( sub6Train );
max_value(1) = max(abs_value(:));
if len > 1
    abs_value = abs( highTrain );
    max_value(2) = max(abs_value(:));
    abs_value = abs(highTrainDoD_phi);
    max_value(3) = max(abs_value);
    abs_value = abs(highTrainDoD_theta);
    max_value(4) = max(abs_value);
    abs_value = abs(highTrainDoA_phi);
    max_value(5) = max(abs_value);
    abs_value = abs(highTrainDoA_theta);
    max_value(6) = max(abs_value);
    abs_value = abs(highTrainPhase);
    max_value(7) = max(abs_value);
    abs_value = abs(highTrainToA);
    max_value(8) = max(abs_value);
    abs_value = abs(highTrainPower);
    max_value(9) = max(abs_value);
end
options.dataStats = max_value;

%------------------------------------------------------
% Prepare inputs:
% ---------------
%Probe for sanity check
temp.sub6Train = sub6Train;
temp.dataStats = options.dataStats(1);
sub6Train = sub6Train/options.dataStats(1);% normalize training data
sub6Val = sub6Val/options.dataStats(1);% normalize validation data
X = zeros(options.numAnt(1),options.numSub,2,options.numOfTrain);
Y = zeros(options.numAnt(1),options.numSub,2,options.numOfVal);
for i = 1:options.numOfTrain
    X(:,:,1,i) = abs(sub6Train(:,:,i));
    X(:,:,2,i) = angle(sub6Train(:,:,i));
end
for i = 1:options.numOfVal
    Y(:,:,1,i) = abs(sub6Val(:,:,i));
    Y(:,:,2,i) = angle(sub6Val(:,:,i));
end
if options.noisyInput
    % Noise power
    NF=5;% Noise figure at the base station
    Pr=30;
    BW=options.bandWidth_sub6*1e9; % System bandwidth in Hz
    noise_power_dB=-204+10*log10(BW/options.numSub)+NF; % Noise power in dB
    noise_power=10^(.1*(noise_power_dB));% Noise power
    fprintf("Sub-6 Noise power is");
    disp(noise_power);  
    % Pilot based channel estimation consideration, divide noise power by
    % pilot power
    pilot_power_dB = options.pilot_power_sub6_dB;
    pilot_power = 10^(.1*(pilot_power_dB));
    fprintf("Sub-6 Pilot power is");
    disp(pilot_power);  
    Pn_r=(noise_power/(pilot_power*((options.dataStats(1))^2)))/2 ;   % Change index to feature_ind
    %Pn=Pn_r/(10^(.1*(options.transPower-Pr)));              %TODO why???
    Pn = Pn_r;                                               %TODO Modify
    % Adding noise
    fprintf(['Corrupting channel measurements with ' num2str(Pn) '-variance Gaussian\n'])
    noise_samples = sqrt(Pn)*randn(size(X));% Zero-mean unity-variance noise
    %Probe into values for sanity check
    temp.noise_samples = noise_samples;
    temp.X = X;
    temp.Y = Y;
    save ("temp.mat","-struct","temp");
    X = X + noise_samples;
    noise_samples = sqrt(Pn)*randn(size(Y));
    Y = Y + noise_samples;
    
else
    fprintf('Clean channel measurements')
end
dataset.inpTrain = X;
dataset.inpVal = Y;

%-----------------------------------------------------
% Prepare outputs:
% ----------------


highTrain = highTrain(1:options.numAnt(2),1:options.numSub,:)/options.dataStats(2);
highVal = highVal(1:options.numAnt(2),1:options.numSub,:)/options.dataStats(2);

%% Corrupting highTrain using pilot based noise (it makes sense to corrupt highTrain and not X, because beamformer indices are being calculated using highTrain directly)
if options.noisyInput
    % Noise power
    NF=0;% Noise figure at the base station
    Pr=30;
    BW=options.bandWidth_THz*1e9; % System bandwidth in Hz
    noise_power_dB=-204+10*log10(BW/options.numSub)+NF; % Noise power in dB
    noise_power=10^(.1*(noise_power_dB));% Noise power
    fprintf("THz Noise power is");
    disp(noise_power);  
    % Pilot based channel estimation consideration, divide noise power by
    % pilot power
    pilot_power_dB = options.pilot_power_THz_dB;
    pilot_power = 10^(.1*(pilot_power_dB));
    fprintf("THz Pilot power is");
    disp(pilot_power);  
    Pn_r=(noise_power/(pilot_power*((options.dataStats(2))^2)))/2 ;   % Change index to feature_ind
    %Pn=Pn_r/(10^(.1*(options.transPower-Pr)));              %TODO why???
    Pn = Pn_r;                                               %TODO Modify
    % Adding noise
    fprintf(['Corrupting channel measurements with ' num2str(Pn) '-variance Gaussian\n'])
    complex_noise_samples = sqrt(Pn)*randn(size(highTrain))+j*sqrt(Pn)*randn(size(highTrain));% Zero-mean unity-variance complex noise
    % Sanity check
    temp_bomb.complex_noise_samples = complex_noise_samples;
    temp_bomb.highTrain = highTrain;
    save ("temp_bomb.mat","-struct","temp_bomb");
    highTrain = highTrain + complex_noise_samples;
end

% X = zeros(1,1,2*options.outputDim,options.numOfTrain);
% Y = zeros(1,1,2*options.outputDim,options.numOfVal);
X = zeros(1,1,2*options.outputDim,options.numOfTrain);
Y = zeros(1,1,2*options.outputDim,options.numOfVal);
for i = 1:options.numOfTrain
    x = highTrain(:,:,i);
    x = x(:, 1:options.numSub);
    %x = [real(x);imag(x)];
    x = [abs(x);angle(x)];
    X(1,1,:,i) = reshape(x, [numel(x),1]);
end
for i = 1:options.numOfVal
    y = highVal(:,:,i);
    y = y(:, 1:options.numSub);
    %y = [real(y);imag(y)];
    y = [abs(y);angle(y)];
    Y(1,1,:,i) = reshape(y, [numel(y),1]);
end
dataset.outpTrain = X;
dataset.outpVal = Y;

%Putting output channel parameters in dataset
dataset.outpTrainDoD_phi = reshape(highTrainDoD_phi/options.dataStats(3),[1,1,1,options.numOfTrain]);
dataset.outpValDoD_phi = reshape(highValDoD_phi/options.dataStats(3),[1,1,1,options.numOfVal]);
dataset.outpTrainDoD_theta = reshape(highTrainDoD_theta/options.dataStats(4),[1,1,1,options.numOfTrain]);
dataset.outpValDoD_theta = reshape(highValDoD_theta/options.dataStats(4),[1,1,1,options.numOfVal]);
dataset.outpTrainDoA_phi = reshape(highTrainDoA_phi/options.dataStats(5),[1,1,1,options.numOfTrain]);
dataset.outpValDoA_phi = reshape(highValDoA_phi/options.dataStats(5),[1,1,1,options.numOfVal]);
dataset.outpTrainDoA_theta = reshape(highTrainDoA_theta/options.dataStats(6),[1,1,1,options.numOfTrain]);
dataset.outpValDoA_theta = reshape(highValDoA_theta/options.dataStats(6),[1,1,1,options.numOfVal]);
dataset.outpTrainPhase = reshape(highTrainPhase/options.dataStats(7),[1,1,1,options.numOfTrain]);
dataset.outpValPhase = reshape(highValPhase/options.dataStats(7),[1,1,1,options.numOfVal]);
dataset.outpTrainToA = reshape(highTrainToA/options.dataStats(8),[1,1,1,options.numOfTrain]);
dataset.outpValToA = reshape(highValToA/options.dataStats(8),[1,1,1,options.numOfVal]);
dataset.outpTrainPower = reshape(highTrainPower/options.dataStats(9),[1,1,1,options.numOfTrain]);
dataset.outpValPower = reshape(highValPower/options.dataStats(9),[1,1,1,options.numOfVal]);

%Putting all parameters together as matrix for join prediction
dataset.outpTrainChannelParams = cat(3,dataset.outpTrainDoD_phi,dataset.outpTrainDoD_theta,dataset.outpTrainDoA_phi, ...
    dataset.outpTrainDoA_theta,dataset.outpTrainPhase,dataset.outpTrainToA,dataset.outpTrainPower);
dataset.outpValChannelParams = cat(3,dataset.outpValDoD_phi,dataset.outpValDoD_theta,dataset.outpValDoA_phi, ...
    dataset.outpValDoA_theta,dataset.outpValPhase,dataset.outpValToA,dataset.outpValPower);

dataset.highFreqChTrain = highTrain;% 
dataset.highFreqChVal = highVal;% 
W = options.codebook;
value_set = 1:size(W,2);
for i = 1:options.numOfTrain
    H = highTrain(:,:,i);
    rec_power = abs( H'*W ).^2;
    %rate_per_sub = log2( 1 + rec_power);
    rate_per_sub = log2( 1 + rec_power * options.transSNR);
    rate_ave = sum(rate_per_sub,1)/options.numSub;
    [r,ind] = max( rate_ave, [], 2 );
    beam_ind(i,1) = ind;
    max_rate(i,1) = r;
end
dataset.labelTrain = categorical( beam_ind, value_set );
dataset.maxRateTrain = max_rate;
beam_ind = [];
max_rate = [];
for i = 1:options.numOfVal
    H = highVal(:,:,i);
    rec_power = abs( H'*W ).^2;
    %rate_per_sub = log2( 1 + rec_power );
    rate_per_sub = log2(1 + rec_power * options.transSNR);
    rate_ave = sum(rate_per_sub,1)/options.numSub;
    [r,ind] = max( rate_ave, [], 2 );
    beam_ind(i,1) = ind;
    max_rate(i,1) = r;
end
dataset.labelVal = categorical( beam_ind, value_set );
dataset.maxRateVal = max_rate;
dataset = rmfield(dataset,'data');
dataset = rmfield(dataset,'labels');
%dataset = rmfield(dataset, 'userLoc');