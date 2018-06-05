% function [C,md1,md2,md3,md4,md5] = train()
    %% Find SIFT features of all the images in the dataset
    dirFiles = dir('images/');
    idx = 1;sz = 0;img_cnt = 0;
    for j = 1:size(dirFiles,1)
        if ~strcmp(dirFiles(j).name,".") && ~strcmp(dirFiles(j).name,"..")
            dirFiles(j).name
            srcFiles = dir(strcat('images/',dirFiles(j).name,'/*.jpg'));
            for i = 1 : round(0.6*size(srcFiles,1))
                filename = strcat('images/',dirFiles(j).name,'/',srcFiles(i).name);
                img = imread(filename);
                I = single(rgb2gray(img));
                [~,d] = vl_sift(I');
                desc(sz+1:sz+size(d',1),:) = d';
                sz = sz+size(d',1);
                img_cnt = img_cnt + 1;
                labels(img_cnt) = idx;
            end
            idx = idx + 1;
        end
    end
    labels = labels';

    %% Use Kmeans to cluster them into cNumber of clusters
    no_clusters = 500;
    [group_no,C] = kmeans(double(desc),no_clusters);
    
    %% Find normalized feature vector for each image
    % md1 = fitcknn(double(desc),group_no,'NumNeighbors',3,'Standardize',1);
    bin = zeros(img_cnt,no_clusters);
    idx = 1;
    for j = 1:size(dirFiles,1)
        if ~strcmp(dirFiles(j).name,".") && ~strcmp(dirFiles(j).name,"..")
            dirFiles(j).name
            srcFiles = dir(strcat('images/',dirFiles(j).name,'/*.jpg'));
            for i = 1 : round(0.6*size(srcFiles,1))
                filename = strcat('images/',dirFiles(j).name,'/',srcFiles(i).name);
                img = imread(filename);
                I = single(rgb2gray(img));
                [~,d] = vl_sift(I');
                x = knnsearch(C,double(d'));
                % x = predict(md1,double(d'));
                uv = unique(x);
                n  = histc(x,uv);
                bin(idx,uv) = n;
                if sum(bin(idx,:)) ~= 0 
                    bin(idx,:) = bin(idx,:)/sum(bin(idx,:));
                end
                idx = idx + 1;
            end
        end
    end
    
    %% Train using svm 
    md1 = fitcsvm([bin(labels == 1,:);bin(labels == 2,:)],[labels(labels == 1);labels(labels == 2)],'Standardize',true,'KernelFunction','RBF','KernelScale','auto')
    md2 = fitcsvm([bin(labels == 1,:);bin(labels == 3,:)],[labels(labels == 1);labels(labels == 3)],'Standardize',true,'KernelFunction','RBF','KernelScale','auto')
    md3 = fitcsvm([bin(labels == 1,:);bin(labels == 4,:)],[labels(labels == 1);labels(labels == 4)],'Standardize',true,'KernelFunction','RBF','KernelScale','auto')
    md4 = fitcsvm([bin(labels == 1,:);bin(labels == 5,:)],[labels(labels == 1);labels(labels == 5)],'Standardize',true,'KernelFunction','RBF','KernelScale','auto')
    md5 = fitcsvm([bin(labels == 2,:);bin(labels == 3,:)],[labels(labels == 2);labels(labels == 3)],'Standardize',true,'KernelFunction','RBF','KernelScale','auto')
    md6 = fitcsvm([bin(labels == 2,:);bin(labels == 4,:)],[labels(labels == 2);labels(labels == 4)],'Standardize',true,'KernelFunction','RBF','KernelScale','auto')
    md7 = fitcsvm([bin(labels == 2,:);bin(labels == 5,:)],[labels(labels == 2);labels(labels == 5)],'Standardize',true,'KernelFunction','RBF','KernelScale','auto')
    md8 = fitcsvm([bin(labels == 3,:);bin(labels == 4,:)],[labels(labels == 3);labels(labels == 4)],'Standardize',true,'KernelFunction','RBF','KernelScale','auto')
    md9 = fitcsvm([bin(labels == 3,:);bin(labels == 5,:)],[labels(labels == 3);labels(labels == 5)],'Standardize',true,'KernelFunction','RBF','KernelScale','auto')
    md10 = fitcsvm([bin(labels == 4,:);bin(labels == 5,:)],[labels(labels == 4);labels(labels == 5)],'Standardize',true,'KernelFunction','RBF','KernelScale','auto')
% end