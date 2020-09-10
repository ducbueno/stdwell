y_matlab = readmatrix("y_.txt");
y_opencl = readmatrix("y_-opencl.txt");

figure
hold on
plot(y_matlab, 'g');
plot(y_opencl, 'b');
plot(y_matlab - y_opencl, 'r');
hold off
legend('matlab', 'opencl', 'difference');