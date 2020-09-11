dim = 3;
dim_wells = 4;
vals_per_block = dim*dim_wells;

model = 'spe10model1/';
dir = strcat('real/', model);
rowptr = readmatrix(strcat(dir, 'val_pointers.txt'));
Ccols = readmatrix(strcat(dir, 'Ccols.txt'));
Bcols = readmatrix(strcat(dir, 'Bcols.txt'));
Cnnzs = readmatrix(strcat(dir, 'Cnnzs.txt'));
Dnnzs = readmatrix(strcat(dir, 'Dnnzs.txt'));
Bnnzs = readmatrix(strcat(dir, 'Bnnzs.txt'));
x = readmatrix(strcat(dir, 'x.txt'));
y = readmatrix(strcat(dir, 'y.txt'));

num_std_wells = length(rowptr) - 1;

B = zeros(dim_wells*num_std_wells, length(x));
for row = 1:num_std_wells
    rrow = 4*row - 3;
    num_blocks = rowptr(row + 1) - rowptr(row);
    for block = 1:num_blocks     
        current_block = block + rowptr(row);
        B(rrow:(rrow + 3), (3*Bcols(current_block) + 1):(3*Bcols(current_block) + 3)) = ...
            reshape(Bnnzs(((current_block - 1)*vals_per_block + 1):(current_block*vals_per_block)), [dim_wells, dim]);
    end
end

D = zeros(dim_wells*num_std_wells);
for idx = 1:num_std_wells
    ridx = 4*idx - 3;
    D(ridx:(ridx + 3), ridx:(ridx + 3)) = reshape(Dnnzs(((idx - 1)*16 + 1):idx*16), [dim_wells, dim_wells]);
end

C = zeros(dim_wells*num_std_wells, length(x));
for row = 1:num_std_wells
    rrow = 4*row - 3;
    num_blocks = rowptr(row + 1) - rowptr(row);
    for block = 1:num_blocks     
        current_block = block + rowptr(row);
        C(rrow:(rrow + 3), (3*Ccols(current_block) + 1):(3*Ccols(current_block) + 3)) = ...
            reshape(Cnnzs(((current_block - 1)*vals_per_block + 1):(current_block*vals_per_block)), [4, 3]);
    end
end

Bx = B*x;
DBx = D*Bx;
y_ = y - C'*DBx;

writematrix(y_, strcat(dir, 'y_-matlab.txt'))