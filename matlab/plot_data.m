function plot_data(model)
    y = readmatrix(strcat('../data/real/', model,'/y.txt'));
    y_opencl = readmatrix(strcat('../data/real/', model,'/y_-opencl.txt'));
    y_cuda = readmatrix(strcat('../data/real/', model,'/y_-cuda.txt'));
    
    err_opencl = norm(y_cuda - y_opencl)/norm(y_cuda);
    
    figure; 
    subplot(1, 2, 1);
    plot(y_cuda - y);
    title('CUDA');
    subplot(1, 2, 2);
    plot(y_opencl - y);
    title('OpenCL');
    sgtitle(strcat({'Model '}, upper(model), {', error = '}, num2str(err_opencl)));
end
