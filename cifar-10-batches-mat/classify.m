batch1 = load('cifar-10-batches-mat/data_batch_1.mat');
batch2 = load('cifar-10-batches-mat/data_batch_2.mat');
batch3 = load('cifar-10-batches-mat/data_batch_3.mat');
batch4 = load('cifar-10-batches-mat/data_batch_4.mat');
batch5 = load('cifar-10-batches-mat/data_batch_5.mat');

classes = [0,3];
images = [batch1.data(logical(sum(batch1.labels == classes,2)),:);batch2.data(logical(sum(batch2.labels == classes,2)),:);batch3.data(logical(sum(batch3.labels == classes,2)),:);batch4.data(logical(sum(batch4.labels == classes,2)),:);batch5.data(logical(sum(batch5.labels == classes,2)),:);];
lables = [batch1.labels(logical(sum(batch1.labels == classes,2)));batch2.labels(logical(sum(batch2.labels == classes,2)));batch3.labels(logical(sum(batch3.labels == classes,2)));batch4.labels(logical(sum(batch4.labels == classes,2)));batch5.labels(logical(sum(batch5.labels == classes,2)));];
[C,no_clusters,md2] = train(images,lables);

t_batch = load('test_batch.mat');
images = t_batch.data(logical(sum(t_batch.labels == classes,2)),:);
lables = t_batch.labels(logical(sum(t_batch.labels == classes,2)));
acc = test(images,lables,C,no_clusters,md2);