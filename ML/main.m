%================================================================================%
% Main script for training and testing a DL model to predict mmWave (28   
% GHz) beam indecies from sub-6 GHz channels. The script assumes the data
% provided in the form of two (.mat) files: 
%   - dataFile1: Sub-6 data
%   - dataFile2: 28GHz data
% Each .mat is a data structure with the following fields: channels and user 
% locations. Channels should be arranged into a 3D array with the following 
% dimensions: # of antenna elements X # of sub-carriers X # of users. User 
% locations should be a 2D array with the folllowing dimensions: 3 X # of users. 
%  The "options" structure provide control over the type of experiment to run.
% -------------------------------------------------------------------------------
% Author: 
% Muhammad Alrabeiah,
% Sept 2019.
%
%=================================================================================%
clc
clear
close all

tx_power = [-42.3375, -37.3375  -32.3375  -27.3375  -22.3375  -17.3375  -12.3375   -7.3375, -2.3375];
% %tx_power = [-17.3375  -12.3375  -7.3375  -2.3375  2.6625  7.6625  12.6625];
snr_db = [-15.0586, -10.0586   -5.0586   -0.0586    4.9414    9.9414   14.9414   19.9414, 24.9414];
% tx_power = [-7.3375];
% snr_db = [19.9414];
snr = 10.^(0.1*snr_db);
error_params=[];
num_ant = [4];% number of sub-6 antennas
num_ant_mm = [64];
accuracy_top1 = zeros(length(num_ant_mm), length(tx_power));
accuracy_top3 = zeros(length(num_ant_mm), length(tx_power));
ave_rate_top1 = zeros(length(num_ant_mm), length(tx_power));
ave_rate_top3 = zeros(length(num_ant_mm), length(tx_power));
ave_upper = zeros(length(num_ant_mm), length(tx_power));
options.figCount = 0;
options.type = 'MLP1';
options.case = 'LOS';
options.expTag = [options.type '_' options.case '_variableSNR'];
options.top_n = 10;
options.valAccuracy = zeros(length(num_ant_mm),length(tx_power));
options.normMethod = 'perDataset';
options.gpuInd = 1;
fprintf('Experiment: %s\n', options.expTag);

for ant = 1:length(num_ant_mm)% number of antennas to loop over
	fprintf('Number of sub-6 antennas: %d and number of mmWave antennas: %d\n', num_ant(1), num_ant_mm(ant))
    [W,~] = UPA_codebook_generator(1,num_ant_mm(ant),1,1,1,1,0.5);% Beam codebook
	options.codebook = W;
	options.numAnt = [num_ant(1), num_ant_mm(ant)];
	options.numSub = 64;                                                 %TODO MODIFY
	options.inputDim = options.numSub*options.numAnt(1);
    options.outputDim = options.numSub*options.numAnt(2);                %For channel matrix outputs only
	options.valPer = 0.3;
	options.inputSize = [1,1,2*options.inputDim];
    options.outputSize = [1,1,1];
	options.noisyInput = true;                                          %TODO MODIFY
	options.bandWidth = 0.02;   %TODO MODIFY
	options.dataFile1 ='DataStructures/2p4GHz.mat';% The path to the sub-6 data file
    options.dataFile2 ='DataStructures/100GHz.mat';% The path to the mmWave data file
    if isempty(options.dataFile1)
		error('Please provide a sub-6 data file!');
	elseif isempty(options.dataFile2)
		error('Please provide a mmWave data file');
	end
	% training settings
	%options.solver = 'sgdm';
    options.solver = 'sgdm';
	%options.learningRate = 1e-1;
    %options.learningRate = 1e-3; %Apt for DoD_phi prediction
    options.learningRate = 1e-4;
    %options.momentum = 0.9;
	options.momentum = 0.9;
	options.schedule = 800;
	options.dropLR = 'piecewise';
	options.dropFactor = 0.1;
	options.maxEpoch = 10;        %TODO MODIFY
	options.batchSize = 100;       %TODO MODIFY
	options.verbose = 1;    
	options.verboseFrequency = 1;
    options.valFreq = 10;           %TODO MODIFY
	options.shuffle = 'every-epoch';
	%options.weightDecay = 1e-4;
    options.weightDecay = 1e-4;
	options.progPlot = 'none';
	for p = 1:length(tx_power)

        fprintf('Pt = %4.2f (dBm)\n', tx_power(p))

		% Prepare dataset:
		% ----------------

		options.transPower = tx_power(p);
        options.transSNR = snr(p);
		fileName = struct('name',{options.dataFile1,...
		                          options.dataFile2});
%         if ~exist('./dataprep_dataset.mat','file')
%             fprintf ("Preparing dataset");
%  		    [dataset,options] = dataPrep(fileName, options);% dataset is a 1x2 structure array
%             save('./dataprep_dataset.mat','dataset');
%             save('./dataprep_options.mat','options');
%         else
%             fprintf("Loading old dataset");
%             load('./dataprep_dataset.mat','dataset');
%             load('./dataprep_options.mat','options');
%             options.maxEpoch = 10;
%             options.learningRate = 1e-4;
%             options.transPower = tx_power(p);
%             options.transSNR = snr(p);
%         end
            fprintf ("Preparing dataset");
  		    [dataset,options] = dataPrep(fileName, options);% dataset is a 1x2 structure array
            save('./dataprep_dataset.mat','dataset');
            save('./dataprep_options.mat','options');

		% Build network:
		% --------------

 		% net = buildNet(options);
        % analyzeNetwork(net)

		% Train network:
		% --------------

% 		trainingOpt = trainingOptions(options.solver, ...
% 		    'InitialLearnRate',options.learningRate,...
% 		    'LearnRateSchedule',options.dropLR, ...
% 		    'LearnRateDropFactor',options.dropFactor, ...
% 		    'LearnRateDropPeriod',options.schedule, ...
% 		    'MaxEpochs',options.maxEpoch, ...
% 		    'L2Regularization',options.weightDecay,...
% 		    'Momentum',options.momentum,...
% 		    'Shuffle', options.shuffle,...
% 		    'MiniBatchSize',options.batchSize, ...
% 		    'ValidationData', {dataset(1).inpVal, dataset(1).outpValDoD_phi},...
%             'ValidationFrequency', options.valFreq,...
% 		    'Verbose', options.verbose,...
% 		    'verboseFrequency', options.verboseFrequency,...
% 		    'Plots',options.progPlot);
% 
% 		gpuDevice(options.gpuInd)
%  		[trainedNet, trainInfo] = trainNetwork(dataset.inpTrain, dataset.outpTrainDoD_phi, net, trainingOpt);
		%options.valAccuracy(ant,p) = trainInfo.ValidationAccuracy(end);
%         if ~exist('./trained_model.mat','file')
%             save ('./trained_model.mat','trainedNet');
%         else
%             load('./trained_model.mat','trainedNet');
%         end 
		% Test network:
		% -------------

 		X = dataset.inpVal;
%         %Y = single( dataset.labelVal );
        %pred = trainedNet.predict(X);

        

%         %Convert pred back to a channel params
%         pred = reshape(pred,[options.outputSize(3),options.numOfVal]);
%         %Denormalization
%         channel_params_datastats = options.dataStats(3);
%         channel_params_datastats = reshape(channel_params_datastats,[length(channel_params_datastats),1]);
%         pred = channel_params_datastats .* pred;
% 
%         %Storing channel_params error
%         curr_error_params = zeros([1,options.outputSize(3)]);
%         for param=1:options.outputSize(3)
%             maxerror=-999;
%             sumerror=0;
%             for user=1:options.numOfVal
%                  curr_error = abs(pred(param,user) - options.dataStats(3)*dataset.outpValChannelParams(1,1,1,user));
% %                curr_error = 1.0/100*options.dataStats(9) * abs(pred(param,user) - dataset.outpValChannelParams(1,1,7,user));
%                 if curr_error>maxerror
%                     maxerror = curr_error;
%                 end
%                 sumerror = sumerror + curr_error;
%             end
%             avg_error = sumerror/options.numOfVal;
%             curr_error_params(param) = avg_error;
%         end
%         error_params = [error_params;curr_error_params];

        %If individual feature predicted files are present, concatenate
        if exist('./pred_DoD_phi.mat','file')
            load('./pred_DoD_phi.mat');
            DoD_phi = pred;
            load('./pred_DoD_theta.mat');
            DoD_theta = pred;
            load('./pred_DoA_phi.mat');
            DoA_phi = pred;
            load('./pred_DoA_theta.mat');
            DoA_theta = pred;
            load('./pred_Phase.mat');
            Phase = pred;
            load('./pred_ToA.mat');
            ToA = pred;
            load('./pred_Power.mat');
            Power = pred;
            pred = [DoD_phi;DoD_theta;DoA_phi;DoA_theta;Phase;ToA;Power];
        end    

        %Computing H_pred from predicted channel params
        %%%%% Computing channel parameter arrays
        CIR = [options.numOfVal];
        DoD = [options.numOfVal];
        DoA = [options.numOfVal];
        curr_paths = 0;
        curr = 1;
        LOS = [options.numOfVal];
        for i = 1:options.numOfVal
            num_paths_max = 1;
            num_paths = 0;
            CIR = [CIR i 0];
            DoD = [DoD i 0];
            DoA = [DoA i 0];
            curr = curr + 2;
            while num_paths<num_paths_max
                try
                    if (num_paths==0)
                        LOS = [LOS 1];
                    end
                    phase = pred(5,i);
                    ToA = pred(6,i);
                    recv_power = pred(7,i);
                    departure_azi = pred(1,i);
                    departure_ele = pred(2,i);
                    arrival_azi = pred(3,i);
                    arrival_ele = pred(4,i);
                    num_paths = num_paths + 1;
                    CIR = [CIR num_paths phase ToA recv_power];
                    DoD = [DoD num_paths departure_azi departure_ele 9999];
                    DoA = [DoA num_paths arrival_azi arrival_ele 9999];
                    %fprintf ('Happening')
                catch
                    break
                end
            end
            CIR(1,curr) = num_paths;
            DoD(1,curr) = num_paths;
            DoA(1,curr) = num_paths;
            curr = curr + 4 * num_paths;
        end

        %%%%% Storing them in files for subsequent ingestion
        if ~exist('../Raytracing_scenarios' , 'dir')
            fprintf("Creating Raytracing Scenarios directory\n")
            mkdir '../Raytracing_scenarios'
        end
        if exist('../Raytracing_scenarios/O1_100_pred', 'dir')
            fprintf("Deleting old O1_100_pred directory\n")
            rmdir '../Raytracing_scenarios/O1_100_pred' s
        end
        fprintf("Creating new O1_100_pred directory\n")
        mkdir '../Raytracing_scenarios/O1_100_pred'
        save('../Raytracing_scenarios/O1_100_pred/O1_100_pred.1.CIR.mat', 'CIR');
        save('../Raytracing_scenarios/O1_100_pred/O1_100_pred.1.DoD.mat', 'DoD');
        save('../Raytracing_scenarios/O1_100_pred/O1_100_pred.1.DoA.mat', 'DoA');
        save('../Raytracing_scenarios/O1_100_pred/O1_100_pred.1.LoS.mat', 'LOS');

        %%%%% Copy params file from older directory to here
        copyfile('../Raytracing_scenarios/O1_100/O1_100.params.mat','../Raytracing_scenarios/O1_100_pred/O1_100_pred.params.mat');

        %%%%% Creating and renaming deepMIMO dataset from predicted parameters
        run('../DeepMIMO_Dataset_Generator_100GHz_pred.m');
        if exist('../DeepMIMO_dataset/dataset_100GHz_pred.mat')
            delete('../DeepMIMO_dataset/dataset_100GHz_pred.mat');
        end
        copyfile('../DeepMIMO_dataset/dataset_1.mat', '../DeepMIMO_dataset/dataset_100GHz_pred.mat');
        delete('../DeepMIMO_dataset/dataset_1.mat');

        %%%%% Creating H_pred
        run('./datastructure_generator_pred.m');
        load('./DataStructures/100GHz_pred.mat');
        H_pred = rawData.channel;

        %pred = single( pred );
        highFreqCh = dataset.highFreqChVal;
        hit = 0;
        for user = 1:size(X,4)
%             % Top-1 average rate
             H = highFreqCh(:,:,user);
%             w = W(:,pred(user));
%             rec_power = abs( H'*w ).^2;
%             %rate_per_sub = log2( 1 + rec_power );
%             rate_per_sub = log2( 1 + rec_power * snr(p));
%             rate_top1(user) = sum(rate_per_sub)/options.numSub;

            % Top-3 accuracy
            rec_power = abs( H_pred(:,:,user)'*W ).^2;
            %rate_per_sub = log2( 1 +rec_power );
            rate_per_sub =  log2( 1 + rec_power * snr(p));
            ave_rate_per_beam = mean( rate_per_sub, 1);
            [~,ind] = max(ave_rate_per_beam);% the best beam
            [~,sort_ind] = sort( ave_rate_per_beam , 'descend' );
            ten_best_beams = sort_ind(1:10);
            three_best_beams = sort_ind(1:3);
            two_best_beams = sort_ind(1:2);
            if any( three_best_beams == ind )
                hit = hit + 1;
            end

            % Top-1 average rate
            w = W(:,ind);
            rec_power = abs( H'*w ).^2;
            %rate_per_sub = log2( 1 + rec_power );
            rate_per_sub = log2( 1 + rec_power * snr(p));
            rate_top1(user) = sum(rate_per_sub)/options.numSub;
            % Top-3 average rate
            rec_power = abs( H'*W(:,three_best_beams) ).^2;
            %rate_per_sub = log2( 1+rec_power );
            rate_per_sub = log2( 1 + rec_power * snr(p));
            ave_rate_per_beam = mean(rate_per_sub,1);
            rate_top3(user) = max( ave_rate_per_beam );
            % Top-2 average rate
            rec_power = abs( H'*W(:,two_best_beams) ).^2;
            %rate_per_sub = log2( 1+rec_power );
            rate_per_sub = log2( 1 + rec_power * snr(p));
            ave_rate_per_beam = mean(rate_per_sub,1);
            rate_top2(user) = max( ave_rate_per_beam );
            % Top-10 average rate
            rec_power = abs( H'*W(:,ten_best_beams) ).^2;
            %rate_per_sub = log2( 1+rec_power );
            rate_per_sub = log2( 1 + rec_power * snr(p));
            ave_rate_per_beam = mean(rate_per_sub,1);
            rate_top10(user) = max( ave_rate_per_beam );

        end
%         accuracy_top1(ant,p) = options.valAccuracy(ant,p);
         accuracy_top3(ant,p) = 100*(hit/options.numOfVal);
        ave_rate_top1(ant,p) = mean(rate_top1);
        ave_rate_top3(ant,p) = mean(rate_top3);
        ave_rate_top2(ant,p) = mean(rate_top2);
        ave_rate_top10(ant,p) = mean(rate_top10);
        ave_upper(ant,p) = mean(dataset.maxRateVal);
%         fprintf('Top-1 and Top-3 rates: %5.3f & %5.3f. Upper bound: %5.3f\n', ave_rate_top1(ant,p),ave_rate_top3(ant,p),...
%                      mean( dataset.maxRateVal ) );
%         fprintf('Top-1 and Top-3 Accuracies: %5.3f%% & %5.3f%%\n', accuracy_top1(ant,p),accuracy_top3(ant,p));

 	end

 end

% Save performance variables
variable_name = [options.expTag '_results'];
save(variable_name,'ave_rate_top1','ave_rate_top3','ave_rate_top2','ave_rate_top10','ave_upper','error_params')
options.figCount = options.figCount+1;
fig1 = figure(options.figCount);
plot(snr_db, ave_rate_top1(1,:), '-m',...
     snr_db, ave_rate_top3(1,:), '-c',...
     snr_db, ave_rate_top2(1,:), '-r',...
     snr_db, ave_rate_top10(1,:), '-b',...
     snr_db, ave_upper(1,:), '-.k');
% plot(snr_db, ave_rate_top3(1,:), '-r',...
%      snr_db, ave_upper(1,:), '-.k');
xlabel('SNR (dB)'); 
ylabel('Spectral Efficiency (bits/sec/Hz)');
grid on
legend('Baseline Top-1 rate','Proposed method Top-1 rate','Baseline Top-3 rate','Proposed method Top-3 rate','Extensive search with exact channel (UB) rate')
name_file = ['ansVSrate_' options.expTag];
saveas(fig1,name_file)
