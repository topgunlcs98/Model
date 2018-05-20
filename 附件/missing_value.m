% ���ļ��������3 ���Ƚ���ȱʧֵ����ٽ��лع�Ԥ��

% ����ÿ�е�l2����;�ֵ������
[n,m]=size(train_data);
train_mean=mean(train_data);
train_l2=zeros([n,1]);
for i=1:n,
    train_mean(i)=norm(mean(train_data(i,:)), 2);
end
save 'train_mean.mat' train_mean
save 'train_norm.mat' train_l2

% �ȿ����ʽ
data=load('data.mat');
data=cell2mat(struct2cell(data)); % ת���ɾ���
test_data=data(284001:284727,1:28); % test_data������Ԥ���727�����ݣ����в���������nan����Ҫ�����
train_mean=load('train_mean.mat'); % mean��ÿһ��ָ��ľ�ֵ��������Ԥ�����������ȱʧֵ��ʱ�����þ�ֵ���Ȼ�����ȿ��ķ�ʽ�ٴ��
train_mean=cell2mat(struct2cell(train_mean));
train_l2=load('train_norm.mat'); % train_l2�Ǽ��������ѵ���������е�ÿ����¼��ÿ�е�l2���룩����һ��һά����,
train_l2=cell2mat(struct2cell(train_l2));
train_data=data(1:284000,1:28); % 284001-284727������ģ��ȱʧֵ

origin_test_data=test_data;


% �ҵ�����nan���У����ҵ�����nan��һ������
% ʹ���ȿ��ķ�ʽ�����
[n,m]=size(test_data);
flag=0;% ��һ�У�����һ���������Ƿ���ȱʧֵ
k=0; % ����ͳ��ÿ���е�ȱʧֵ
index=[]; % ÿ���е�ȱʧֵ���±�
error_count=0;% error���±�
error=[]; % ͳ��������е���mse��
for i=1:n,
    for j=1:m,
        if isnan(test_data(i,j)),
            flag=1;
          	k=k+1;
            index(k)=j;
            test_data(i,j)=train_mean(j); %������һ�еľ�ֵ�
            % �����ȿ��ķ�ʽ�Ա������ݽ����,�ȼ���L2���룬��28�������ݽ��жԱȵõ���������Ǹ���Ȼ�����Ǹ���ֵ������
        end
    end
    if flag==1, % �ڼ����һ������֮ǰҪ��k��Ϊ0��index��Ϊ��
    	flag=0; % ���flag
        l2=norm(test_data(i,:),2);
        [x,minimum]=min(abs(train_l2-l2));% minimum��28���������l2�������������С���±�
        for z=index,
            test_data(i,z)=train_data(minimum,z);
            error_count=error_count+1;
            error(error_count)=(origin_test_data(i,j)-test_data(i,j))^2;
        end
        k=0;
        index=[];
    end
end


% ���������֤���Ȳ�����ȱ����ֵ��Ȼ�����
% �����������test_data�Ĵ�Сһ�£��������Ϊ���������������ɲ�ȱ�Ĳ���

rand_pos=rand([727,28]);% ����727*28����ɢ���ȷֲ�����������󣻲���һ����ֵ��0-1֮���mm*nn����

% ȱʧ10%������,�����test_dataδ������һ��
test_data=origin_test_data;
test_data(rand_pos<=0.05)=nan;

plot(1:727,error) % Ԥ��ֵ����ʵֵ֮��Ĳ��ƽ����ͼ
mean(error) % MSE