%Convert deepMIMO datasets into data structures in the format required for
%ML ingestion
%% sub - 6GHz
file = load('../DeepMIMO_dataset/dataset_2p4GHz.mat');
DeepMIMO_dataset = file.DeepMIMO_dataset;
n_recv = length(DeepMIMO_dataset{1,1}.user);
n_ant = 4;
n_sub = 64;

% Constructing datastructure for each base station
if ~exist('DataStructures','dir')
    mkdir 'DataStructures';
end
for bs = 1:1
    channel = zeros([n_ant,n_sub,n_recv]);
    labels = zeros([1,n_recv]);
    DoD_phi = zeros([1,n_recv]);
    DoD_theta = zeros([1,n_recv]);
    DoA_phi = zeros([1,n_recv]);
    DoA_theta = zeros([1,n_recv]);
    phase = zeros([1,n_recv]);
    ToA = zeros([1,n_recv]);
    power = zeros([1,n_recv]);
    
    for i=1:n_ant
        for j=1:n_sub
            for k=1:n_recv
                channel(i,j,k) = DeepMIMO_dataset{1,bs}.user{1,k}.channel(1,i,j);
            end
        end
    end
    
    for i=1:n_recv
        labels(1,i) = DeepMIMO_dataset{1,bs}.user{1,i}.LoS_status;
        % For DoA, DoD, phase, ToA, power, taking only first (strongest
        % path) as part of datastructure 
        DoA_phi(1,i) = DeepMIMO_dataset{1,bs}.user{1,i}.DoA_phi(1);
        DoA_theta(1,i) = DeepMIMO_dataset{1,bs}.user{1,i}.DoA_theta(1);
        DoD_phi(1,i) = DeepMIMO_dataset{1,bs}.user{1,i}.DoD_phi(1);
        DoD_theta(1,i) = DeepMIMO_dataset{1,bs}.user{1,i}.DoD_theta(1);
        phase(1,i) = DeepMIMO_dataset{1,bs}.user{1,i}.phase(1);
        ToA(1,i) = DeepMIMO_dataset{1,bs}.user{1,i}.ToA(1);
        power(1,i) = DeepMIMO_dataset{1,bs}.user{1,i}.power(1);
    end
    
    %Constructing data structure
    rawData.channel = channel;
    rawData.labels = labels;
    rawData.DoD_phi = DoD_phi;
    rawData.DoD_theta = DoD_theta;
    rawData.DoA_phi = DoA_phi;
    rawData.DoA_theta = DoA_theta;
    rawData.phase = phase;
    rawData.ToA = ToA;
    rawData.power = power;
    s.rawData = rawData;
    
    if exist(strcat('DataStructures/2p4GHz.',int2str(bs),'.mat'),'file')
        delete (strcat('DataStructures/2p4GHz.',int2str(bs),'.mat'));
    end
    save(strcat('DataStructures/2p4GHz.',int2str(bs),'.mat'),'-struct','s');
end 
%% THz
file = load('../DeepMIMO_dataset/dataset_100GHz.mat');
DeepMIMO_dataset = file.DeepMIMO_dataset;
n_recv = length(DeepMIMO_dataset{1,1}.user);
n_ant = 64;
n_sub = 64;
channel = zeros([n_ant,n_sub,n_recv]);
labels = zeros([1,n_recv]);
DoD_phi = zeros([1,n_recv]);
DoD_theta = zeros([1,n_recv]);
DoA_phi = zeros([1,n_recv]);
DoA_theta = zeros([1,n_recv]);
phase = zeros([1,n_recv]);
ToA = zeros([1,n_recv]);
power = zeros([1,n_recv]);

for i=1:n_ant
    for j=1:n_sub
        for k=1:n_recv
            channel(i,j,k) = DeepMIMO_dataset{1,1}.user{1,k}.channel(1,i,j);
        end
    end
end

%Constructing datastructures for each base station
if ~exist('DataStructures','dir')
    mkdir 'DataStructures';
end

for bs = 1:1
    for i=1:n_recv
        labels(1,i) = DeepMIMO_dataset{1,bs}.user{1,i}.LoS_status;
        DoA_phi(1,i) = DeepMIMO_dataset{1,bs}.user{1,i}.DoA_phi;
        DoA_theta(1,i) = DeepMIMO_dataset{1,bs}.user{1,i}.DoA_theta;
        DoD_phi(1,i) = DeepMIMO_dataset{1,bs}.user{1,i}.DoD_phi;
        DoD_theta(1,i) = DeepMIMO_dataset{1,bs}.user{1,i}.DoD_theta;
        phase(1,i) = DeepMIMO_dataset{1,bs}.user{1,i}.phase;
        ToA(1,i) = DeepMIMO_dataset{1,bs}.user{1,i}.ToA;
        power(1,i) = DeepMIMO_dataset{1,bs}.user{1,i}.power;
    end
    
    %Constructing data structure
    rawData.channel = channel;
    rawData.labels = labels;
    rawData.DoD_phi = DoD_phi;
    rawData.DoD_theta = DoD_theta;
    rawData.DoA_phi = DoA_phi;
    rawData.DoA_theta = DoA_theta;
    rawData.phase = phase;
    rawData.ToA = ToA;
    rawData.power = power;
    s.rawData = rawData;
    
    if exist(strcat('DataStructures/100GHz.',int2str(bs),'.mat'),'file')
        delete (strcat('DataStructures/100GHz.',int2str(bs),'.mat'));
    end
    save(strcat('DataStructures/100GHz.',int2str(bs),'.mat'),'-struct','s');
end 
