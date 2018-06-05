% function out = findCls(images,C,md)
% clearvars -except C md1 md2 md3 md4 md5 md6 md7 md8 md9 md10
% [C,md1,md2,md3,md4,md5] = train()

%% Find the normalized features vector of the images
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
[out(:,1),cost(:,1:2)] = predict(md1,bin);
[out(:,2),cost(:,3:4)] = predict(md2,bin);
[out(:,3),cost(:,5:6)] = predict(md3,bin);
[out(:,4),cost(:,7:8)] = predict(md4,bin);
[out(:,5),cost(:,9:10)] = predict(md5,bin);
[out(:,6),cost(:,11:12)] = predict(md6,bin);
[out(:,7),cost(:,13:14)] = predict(md7,bin);
[out(:,8),cost(:,15:16)] = predict(md8,bin);
[out(:,8),cost(:,17:18)] = predict(md9,bin);
[out(:,10),cost(:,19:20)] = predict(md10,bin);

cst(:,1) = sum(cost(:,[1,3,5,7]),2);
cst(:,2) = sum(cost(:,[2,9,11,13]),2);
cst(:,3) = sum(cost(:,[4,10,15,17]),2);
cst(:,4) = sum(cost(:,[6,12,16,19]),2);
cst(:,5) = sum(cost(:,[8,14,18,20]),2);

[a,out] = max(cst');
sum(labels==out)/size(out',1)
% end