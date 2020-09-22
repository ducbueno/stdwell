function plot_data(model)
    v = readmatrix(strcat('../data/real/', model,'/v.txt'));
    v_opencl = readmatrix(strcat('../data/real/', model,'/v_-opencl.txt'));
    v_cuda = readmatrix(strcat('../data/real/', model,'/v_-cuda.txt'));
    
    t = readmatrix(strcat('../data/real/', model,'/t.txt'));
    t_opencl = readmatrix(strcat('../data/real/', model,'/t_-opencl.txt'));
    t_cuda = readmatrix(strcat('../data/real/', model,'/t_-cuda.txt'));
    
    err_opencl = norm(v_cuda - v_opencl)/norm(v_cuda) + norm(t_cuda - t_opencl)/norm(t_cuda);
    
    figure; 
    subplot(2, 2, 1);
    plot(v_cuda - v);
    title('CUDA (v)');
    subplot(2, 2, 2);
    plot(v_opencl - v);
    title('OpenCL (v)');
    subplot(2, 2, 3);
    plot(t_cuda - t);
    title('CUDA (t)');
    subplot(2, 2, 4);
    plot(t_opencl - t);
    title('OpenCL (t)');
    sgtitle(strcat({'Model '}, upper(model), {', error = '}, num2str(err_opencl)));
end
