function acc = test(images,lables,C,no_clusters,md2)
    %% Find the normalized features vector of the images
    bin = zeros(size(images,1),no_clusters);
    for idx = 1:size(images,1)
        I = single(rgb2gray(reshape(images(idx,:),32,32,3)));
        [~,d] = vl_sift(I');

        x = knnsearch(C,double(d'));
        % x = predict(md1,double(d'));
        uv = unique(x);
        n  = histc(x,uv);
        bin(idx,uv) = n;
        if sum(bin(idx,:)) ~= 0 
            bin(idx,:) = bin(idx,:)/sum(bin(idx,:));
        end
    end
    
    %% Predict based on the trained data
    out = predict(md2,bin);
    acc = sum(out == lables)/size(lables,1);
end