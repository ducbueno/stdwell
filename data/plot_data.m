function plot_data(model)
    y = readmatrix(strcat('real/', model ,'/y.txt'));
    y_matlab = readmatrix(strcat('real/', model ,'/y_-matlab.txt'));
    y_opencl = readmatrix(strcat('real/', model ,'/y_-opencl.txt'));
    y_cuda = readmatrix(strcat('real/', model ,'/y_-cuda.txt'));
    y_flow = readmatrix(strcat('real/', model ,'/y_-flow.txt'));

    figure;
    subplot(1, 2, 1);
    plot(y_cuda - y);
    subplot(1, 2, 2);
    plot(y_opencl - y);


    % err_opencl = norm(y_cuda - y_opencl)/norm(y_cuda);
    % err_flow = norm(y_cuda - y_flow)/norm(y_cuda);
end