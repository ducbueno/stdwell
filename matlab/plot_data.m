function plot_data(model)
    y = readmatrix(strcat('../data/real/', model,'/y.txt'));
    y_opencl = readmatrix(strcat('../data/real/', model,'/y_-opencl.txt'));
    y_cuda = readmatrix(strcat('../data/real/', model,'/y_-cuda.txt'));

    figure;
    subplot(1, 2, 1);
    plot(y_cuda - y);
    subplot(1, 2, 2);
    plot(y_opencl - y);

    err_opencl = norm(y_cuda - y_opencl)/norm(y_cuda);
    disp(err_opencl);
end
