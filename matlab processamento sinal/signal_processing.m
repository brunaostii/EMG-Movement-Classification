%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------- Signal Processor -----------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------- Load Dataset ------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = dlmread('dataset/1/1_raw_data_13-12_22.03.16.txt', '\t', 1,0);
time = data(:,1)/1000;
CH1 = data(:,2);
CH2 = data(:,3);
CH3 = data(:,4);
CH4 = data(:,5);
CH5 = data(:,6);
CH6 = data(:,7);
CH7 = data(:,8);
CH8 = data(:,9);
clear data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------- Load Variables ----------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
R1 = 15;
R2 = 25e3;
R3 = 10e3;
R4 = 10e3;
R5 = 10e3;

Vdc_offset = -2.5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------- Run The Simulation ---------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sim('circuito.slx')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%----------------------------- Plot data --------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1)
subplot(2,1,1)
plot(tout,Vin_CH1,'b')
xlabel('Tempo (s)')
ylabel('Amplitude (V)')
title('Sinal de Entrada do Canal 1')
xlim([0,max(tout)])
subplot(2,1,2)
plot(tout,Vout_CH1,'b')
xlabel('Tempo (s)')
ylabel('Amplitude (V)')
title('Sinal de Saída do Canal 1')
xlim([0,max(tout)])
ylim([0,5])
saveas(gcf,'CH1.png')

figure(2)
subplot(2,1,1)
plot(tout,Vin_CH2,'b')
xlabel('Tempo (s)')
ylabel('Amplitude (V)')
title('Sinal de Entrada do Canal 2')
xlim([0,max(tout)])
subplot(2,1,2)
plot(tout,Vout_CH2,'b')
xlabel('Tempo (s)')
ylabel('Amplitude (V)')
title('Sinal de Saída do Canal 2')
xlim([0,max(tout)])
ylim([0,5])
saveas(gcf,'CH2.png')

figure(3)
subplot(2,1,1)
plot(tout,Vin_CH3,'b')
xlabel('Tempo (s)')
ylabel('Amplitude (V)')
title('Sinal de Entrada do Canal 3')
xlim([0,max(tout)])
subplot(2,1,2)
plot(tout,Vout_CH3,'b')
xlabel('Tempo (s)')
ylabel('Amplitude (V)')
title('Sinal de Saída do Canal 3')
xlim([0,max(tout)])
ylim([0,5])
saveas(gcf,'CH3.png')

figure(4)
subplot(2,1,1)
plot(tout,Vin_CH4,'b')
xlabel('Tempo (s)')
ylabel('Amplitude (V)')
title('Sinal de Entrada do Canal 4')
xlim([0,max(tout)])
subplot(2,1,2)
plot(tout,Vout_CH1,'b')
xlabel('Tempo (s)')
ylabel('Amplitude (V)')
title('Sinal de Saída do Canal 4')
xlim([0,max(tout)])
ylim([0,5])
saveas(gcf,'CH4.png')


figure(5)
subplot(2,1,1)
plot(tout,Vin_CH5,'b')
xlabel('Tempo (s)')
ylabel('Amplitude (V)')
title('Sinal de Entrada do Canal 5')
xlim([0,max(tout)])
subplot(2,1,2)
plot(tout,Vout_CH1,'b')
xlabel('Tempo (s)')
ylabel('Amplitude (V)')
title('Sinal de Saída do Canal 5')
xlim([0,max(tout)])
ylim([0,5])
saveas(gcf,'CH5.png')


figure(6)
subplot(2,1,1)
plot(tout,Vin_CH6,'b')
xlabel('Tempo (s)')
ylabel('Amplitude (V)')
title('Sinal de Entrada do Canal 6')
xlim([0,max(tout)])
subplot(2,1,2)
plot(tout,Vout_CH6,'b')
xlabel('Tempo (s)')
ylabel('Amplitude (V)')
title('Sinal de Saída do Canal 6')
xlim([0,max(tout)])
ylim([0,5])
saveas(gcf,'CH6.png')


figure(7)
subplot(2,1,1)
plot(tout,Vin_CH7,'b')
xlabel('Tempo (s)')
ylabel('Amplitude (V)')
title('Sinal de Entrada do Canal 7')
xlim([0,max(tout)])
subplot(2,1,2)
plot(tout,Vout_CH7,'b')
xlabel('Tempo (s)')
ylabel('Amplitude (V)')
title('Sinal de Saída do Canal 7')
ylim([0,5])
xlim([0,max(tout)])
saveas(gcf,'CH7.png')


figure(8)
subplot(2,1,1)
plot(tout,Vin_CH8,'b')
xlabel('Tempo (s)')
ylabel('Amplitude (V)')
title('Sinal de Entrada do Canal 8')
xlim([0,max(tout)])
subplot(2,1,2)
plot(tout,Vout_CH8,'b')
xlabel('Tempo (s)')
ylabel('Amplitude (V)')
title('Sinal de Saída do Canal 8')
ylim([0,5])
xlim([0,max(tout)])
saveas(gcf,'CH8.png')