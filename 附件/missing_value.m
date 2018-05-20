% 此文件针对问题3 ，先进行缺失值填补，再进行回归预测

% 计算每行的l2距离和均值并保存
[n,m]=size(train_data);
train_mean=mean(train_data);
train_l2=zeros([n,1]);
for i=1:n,
    train_mean(i)=norm(mean(train_data(i,:)), 2);
end
save 'train_mean.mat' train_mean
save 'train_norm.mat' train_l2

% 热卡填补方式
data=load('data.mat');
data=cell2mat(struct2cell(data)); % 转换成矩阵
test_data=data(284001:284727,1:28); % test_data是用于预测的727个数据，其中部分数据是nan，需要进行填补
train_mean=load('train_mean.mat'); % mean是每一个指标的均值，当用于预测的数据中有缺失值的时候先用均值填补，然后用热卡的方式再次填补
train_mean=cell2mat(struct2cell(train_mean));
train_l2=load('train_norm.mat'); % train_l2是计算好用于训练的数据中的每条记录（每行的l2距离），是一个一维向量,
train_l2=cell2mat(struct2cell(train_l2));
train_data=data(1:284000,1:28); % 284001-284727被用于模拟缺失值

origin_test_data=test_data;


% 找到含有nan的行，即找到含有nan的一条数据
% 使用热卡的方式进行填补
[n,m]=size(test_data);
flag=0;% 这一行，即这一条数据中是否有缺失值
k=0; % 用于统计每行中的缺失值
index=[]; % 每行中的缺失值的下标
error_count=0;% error的下标
error=[]; % 统计填补过程中的误差（mse）
for i=1:n,
    for j=1:m,
        if isnan(test_data(i,j)),
            flag=1;
          	k=k+1;
            index(k)=j;
            test_data(i,j)=train_mean(j); %先用这一列的均值填补
            % 利用热卡的方式对本条数据进行填补,先计算L2距离，和28万条数据进行对比得到最相近的那个，然后用那个的值来代替
        end
    end
    if flag==1, % 在检查下一行数据之前要把k置为0，index置为空
    	flag=0; % 清除flag
        l2=norm(test_data(i,:),2);
        [x,minimum]=min(abs(train_l2-l2));% minimum是28万个数据中l2距离和这个相差最小的下标
        for z=index,
            test_data(i,z)=train_data(minimum,z);
            error_count=error_count+1;
            error(error_count)=(origin_test_data(i,j)-test_data(i,j))^2;
        end
        k=0;
        index=[];
    end
end


% 下面进行验证，先产生残缺的数值，然后再填补
% 产生随机数和test_data的大小一致，用这个作为索引，来进行生成残缺的部分

rand_pos=rand([727,28]);% 产生727*28阶离散均匀分布的随机数矩阵；产生一个数值在0-1之间的mm*nn矩阵

% 缺失10%的数据,这里的test_data未经过归一化
test_data=origin_test_data;
test_data(rand_pos<=0.05)=nan;

plot(1:727,error) % 预测值和真实值之间的差的平方作图
mean(error) % MSE