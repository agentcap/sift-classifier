%% Find the normalized features vector of the images

% [C,md1,md2,md3,md4,md5] = train()
% clearvars -except C md1 md2 md3 md4 md5
no_clusters = 500;
idx = 1;
dirFiles = dir('images/');
for j = 1:size(dirFiles,1)
    if ~strcmp(dirFiles(j).name,".") && ~strcmp(dirFiles(j).name,"..")
        dirFiles(j).name
        srcFiles = dir(strcat('images/',dirFiles(j).name,'/*.jpg'));
        for i = round(0.6*size(srcFiles,1)):size(srcFiles,1)
            filename = strcat('images/',dirFiles(j).name,'/',srcFiles(i).name);
            img = imread(filename);
            if size(img,3) == 3
                img = rgb2gray(img);
            end
            I = single(img);
            [~,d] = vl_sift(I');
            x = knnsearch(C,double(d'));
            % x = predict(md1,double(d'));
            uv = unique(x);
            n  = histc(x,uv);
            bin(idx,uv) = n;
            if sum(bin(idx,:)) ~= 0 
                bin(idx,:) = bin(idx,:)/sum(bin(idx,:));
            end
            labels(idx) = j-2;
            idx = idx + 1;
        end
    end
end

%% Predict based on the trained data
[out(:,1),cst] = predict(md1,bin);
cost(:,1) = cst(:,2);
[out(:,2),cst] = predict(md2,bin);
cost(:,2) = cst(:,2);
[out(:,3),cst] = predict(md3,bin);
cost(:,3) = cst(:,2);
[out(:,4),cst] = predict(md4,bin);
cost(:,4) = cst(:,2);
[out(:,5),cst] = predict(md5,bin);
cost(:,5) = cst(:,2);

[~,out] = max(cost');
acc = sum(labels==out)/size(out',1)