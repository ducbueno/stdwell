y_matlab = readmatrix("synth/y_.txt");
y_opencl = readmatrix("synth/y_-opencl.txt");

figure
hold on
plot(y_matlab, 'g');
plot(y_opencl, 'b');
plot(y_matlab - y_opencl, 'r');
hold off
legend('matlab', 'opencl', 'difference');