inner_param = load('./Raytracing_scenarios/O1_100/O1_100.params.mat');
inner_param.carrier_freq = 1.0000e+11;
inner_param.num_BS = 3;
inner_param.transmit_power = 40;
save('./Raytracing_scenarios/O1_100/O1_100.params.mat', '-struct', "inner_param");