y_matlab = readmatrix("real/spe3case1/y_-matlab.txt");
y_opencl = readmatrix("real/spe3case1/y_-opencl.txt");
y_cuda = readmatrix("real/spe3case1/y_-cuda.txt");
y_flow = readmatrix("real/spe3case1/y_.txt");

stru = [y_matlab y_opencl y_cuda y_flow];

% figure
% hold on
% plot(y_matlab, 'g');
% plot(y_opencl, 'b');
% plot(y_cuda, 'r');
% hold off
% legend('opencl', 'cuda');