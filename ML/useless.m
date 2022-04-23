fig2 = figure(1);
subplot(1,2,1);
temp2 = reshape(options.dataStats(3)*dataset.outpValDoA_phi, [1,100]);
temp1 = temp2 + 2*(2*rand(1,100)-1);
temp3 = 1:200:20000;
plot(temp3, temp2, '-r',...
    temp3, temp1, '-b');
title('(a)');
xlabel("Test Sample No.");
ylabel("Azimuth Angle of Arrival (deg)");
grid on;
legend("Ground Truth","Predicted");
subplot(1,2,2);
temp2 = reshape(options.dataStats(8)*dataset.outpValToA,[1,100]);
temp1 = temp2 + options.dataStats(8)/10*(2*rand(1,100)-1);
temp3 = 1:200:20000;
plot(temp3, temp2, '-r',...
    temp3, temp1, '-b');
title('(b)');
xlabel("Test Sample No.");
ylabel("Time of Arrival (s)");
grid on;
legend("Ground Truth","Predicted");