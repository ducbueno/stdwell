% y_matlab = readmatrix("real/spe1case1/y_.txt");
% y_opencl = readmatrix("real/spe1case1/y_-opencl.txt");

figure
hold on
plot(y_, 'g');
plot(y__opencl, 'b');
% plot(y__flow - y__opencl, 'r');
hold off
legend('matlab', 'opencl');