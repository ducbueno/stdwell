dim = 3;
dim_wells = 4;
vals_per_block = dim*dim_wells;
rowptr = [0 2 4 6 8];
cols = [3 33 6 43 9 30 12 48];
Bnnzs = rand(1, 8*vals_per_block);
Dnnzs = rand(1, 4*16);
Cnnzs = rand(1, 8*vals_per_block);

B = zeros(16, 150);
D = zeros(16);
C = zeros(16, 150);

for row = 1:4
    rrow = 4*row - 3;
    num_blocks = rowptr(row + 1) - rowptr(row);
    for block = 1:num_blocks     
        current_block = block + rowptr(row);
        B(rrow:(rrow + 3), (3*cols(current_block) + 1):(3*cols(current_block) + 3)) = ...
            reshape(Bnnzs(((current_block - 1)*vals_per_block + 1):(current_block*vals_per_block)), [4, 3]);
    end
end

for idx = 1:4
    ridx = 4*idx - 3;
    D(ridx:(ridx + 3), ridx:(ridx + 3)) = reshape(Dnnzs(((idx - 1)*16 + 1):idx*16), [4, 4]);
end

for row = 1:4
    rrow = 4*row - 3;
    num_blocks = rowptr(row + 1) - rowptr(row);
    for block = 1:num_blocks     
        current_block = block + rowptr(row);
        C(rrow:(rrow + 3), (3*cols(current_block) + 1):(3*cols(current_block) + 3)) = ...
            reshape(Cnnzs(((current_block - 1)*vals_per_block + 1):(current_block*vals_per_block)), [4, 3]);
    end
end

x = rand(150, 1);
y = rand(150, 1);
 
y_ = y - C'*D*B*x;

writematrix(Bnnzs', 'Bnnzs.txt');
writematrix(Dnnzs', 'Dnnzs.txt');
writematrix(Cnnzs', 'Cnnzs.txt');
writematrix(cols', 'Bcols.txt');
writematrix(cols', 'Ccols.txt');
writematrix(rowptr', 'rowptr.txt');
writematrix(x, 'x.txt');
writematrix(y, 'y.txt');
writematrix(y_, 'y_.txt');