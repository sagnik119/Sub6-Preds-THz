inner_param = load('./Raytracing_scenarios/O1_2p4/O1_2p4.params.mat');
inner_param.carrier_freq = 2.4000e+9;
inner_param.num_BS = 3;
inner_param.transmit_power = 0;
save('./Raytracing_scenarios/O1_2p4/O1_2p4.params.mat', '-struct', "inner_param");