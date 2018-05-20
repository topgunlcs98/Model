% 神经网络过程

% 读取数据
data=load('data.mat'); % 从284262开始 class 是1（违约客户）
data=cell2mat(struct2cell(data)); % 转换为矩阵284727X30
train_data=data(1:284000,1:1:28); % 用于训练回归和分类的数据284000个，其余的727个用于缺失值填补测试
regress_label=data(1:284000,29); % 用于回归的标签

% 使用全部数据训练神经网络
classify_label=data(1:284727,30); % 用于分类的标签，一共有800个 class为0 400个 为1 400个
classify_data=data(1:284727,1:28);


classify_label=data(283796:284727,30); % 用于分类的标签，一共有800个 class为0 400个 为1 400个
classify_data=data(283796:284727,1:28); % 用于分类的数据


% 回归训练数据进行归一化才能输入神经网络
stdr = std(train_data); % 得到标准差
[n, m] = size(train_data); % 矩阵的行与列
train_data=train_data./stdr(ones(n, 1), :); %标准化

% 对用于分类的数据进行归一化
stdr = std(classify_data); % 得到标准差
[n, m] = size(classify_data); % 矩阵的行与列
classify_data=classify_data./stdr(ones(n, 1), :); %标准化

% 进行回归训练的时候需要进行转置
train_data=train_data';
regress_label=regress_label';

% 进行分类的时候需要进行转置
classify_label=classify_label';
classify_data=classify_data';

% 下面是用于回归的测试数据
test_data=data(284001:284727,1:28); % 用于测试的727个数据
test_regress_label=data(284001:284727,29); % 测试额度的标签

% 下面是用于分类的测试数据
test_classify_data0=data(1:100,1:28);
test_classify_label0=data(1:100,30);
test_classify_data1=data(284663:284727,1:28);
test_classify_label1=data(284663:284727,30); % 65个违约的用于预测

% 回归测试数据进行归一化才能输入神经网络
stdr = std(test_data); % 得到标准差
[n, m] = size(test_data); % 矩阵的行与列
test_data=test_data./stdr(ones(n, 1), :); %标准化

% 分类测试数据需要进行归一化才能输入神经网络
stdr = std(test_classify_data0); % 得到标准差
[n, m] = size(test_classify_data0); % 矩阵的行与列
test_classify_data0=test_classify_data0./stdr(ones(n, 1), :); %标准化

stdr = std(test_classify_data1); % 得到标准差
[n, m] = size(test_classify_data1); % 矩阵的行与列
test_classify_data1=test_classify_data1./stdr(ones(n, 1), :); %标准化

% 测试数据进行转置
test_data=test_data';
test_regress_label=test_regress_label';

test_classify_label0=test_classify_label0';
test_classify_data0=test_classify_data0';
test_classify_label1=test_classify_label1';
test_classify_data1=test_classify_data1';


% 建立神经网络中的选项
% Training Function：训练函数
% Adapting learning function：适应性学习函数，即梯度下降的方式
% Performance function：即loss function，
% Number of layers：隐藏层数
% Number of neurons：隐藏层神经元
% Transfer function：激活函数 transig（双极性函数 BP一般使用） LOGSIG（逻辑斯蒂曲线）
% 生成神经网络
% 这个网络一共有四层 每层的激活函数都是tansig（双曲正切S型函数），用于训练的是traingd 梯度下降法 

% 验证回归（信用额度） test_data是已经归一化的结果
net=load('regress_net');
perf=mse(net,test_data,test_regress_label);

% 验证分类（是否违约） test_data是已经归一化的结果
net=load('classifier_net');
pred=sim(net,test_data);
