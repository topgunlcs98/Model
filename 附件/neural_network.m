% ���������

% ��ȡ����
data=load('data.mat'); % ��284262��ʼ class ��1��ΥԼ�ͻ���
data=cell2mat(struct2cell(data)); % ת��Ϊ����284727X30
train_data=data(1:284000,1:1:28); % ����ѵ���ع�ͷ��������284000���������727������ȱʧֵ�����
regress_label=data(1:284000,29); % ���ڻع�ı�ǩ

% ʹ��ȫ������ѵ��������
classify_label=data(1:284727,30); % ���ڷ���ı�ǩ��һ����800�� classΪ0 400�� Ϊ1 400��
classify_data=data(1:284727,1:28);


classify_label=data(283796:284727,30); % ���ڷ���ı�ǩ��һ����800�� classΪ0 400�� Ϊ1 400��
classify_data=data(283796:284727,1:28); % ���ڷ��������


% �ع�ѵ�����ݽ��й�һ����������������
stdr = std(train_data); % �õ���׼��
[n, m] = size(train_data); % �����������
train_data=train_data./stdr(ones(n, 1), :); %��׼��

% �����ڷ�������ݽ��й�һ��
stdr = std(classify_data); % �õ���׼��
[n, m] = size(classify_data); % �����������
classify_data=classify_data./stdr(ones(n, 1), :); %��׼��

% ���лع�ѵ����ʱ����Ҫ����ת��
train_data=train_data';
regress_label=regress_label';

% ���з����ʱ����Ҫ����ת��
classify_label=classify_label';
classify_data=classify_data';

% ���������ڻع�Ĳ�������
test_data=data(284001:284727,1:28); % ���ڲ��Ե�727������
test_regress_label=data(284001:284727,29); % ���Զ�ȵı�ǩ

% ���������ڷ���Ĳ�������
test_classify_data0=data(1:100,1:28);
test_classify_label0=data(1:100,30);
test_classify_data1=data(284663:284727,1:28);
test_classify_label1=data(284663:284727,30); % 65��ΥԼ������Ԥ��

% �ع�������ݽ��й�һ����������������
stdr = std(test_data); % �õ���׼��
[n, m] = size(test_data); % �����������
test_data=test_data./stdr(ones(n, 1), :); %��׼��

% �������������Ҫ���й�һ����������������
stdr = std(test_classify_data0); % �õ���׼��
[n, m] = size(test_classify_data0); % �����������
test_classify_data0=test_classify_data0./stdr(ones(n, 1), :); %��׼��

stdr = std(test_classify_data1); % �õ���׼��
[n, m] = size(test_classify_data1); % �����������
test_classify_data1=test_classify_data1./stdr(ones(n, 1), :); %��׼��

% �������ݽ���ת��
test_data=test_data';
test_regress_label=test_regress_label';

test_classify_label0=test_classify_label0';
test_classify_data0=test_classify_data0';
test_classify_label1=test_classify_label1';
test_classify_data1=test_classify_data1';


% �����������е�ѡ��
% Training Function��ѵ������
% Adapting learning function����Ӧ��ѧϰ���������ݶ��½��ķ�ʽ
% Performance function����loss function��
% Number of layers�����ز���
% Number of neurons�����ز���Ԫ
% Transfer function������� transig��˫���Ժ��� BPһ��ʹ�ã� LOGSIG���߼�˹�����ߣ�
% ����������
% �������һ�����Ĳ� ÿ��ļ��������tansig��˫������S�ͺ�����������ѵ������traingd �ݶ��½��� 

% ��֤�ع飨���ö�ȣ� test_data���Ѿ���һ���Ľ��
net=load('regress_net');
perf=mse(net,test_data,test_regress_label);

% ��֤���ࣨ�Ƿ�ΥԼ�� test_data���Ѿ���һ���Ľ��
net=load('classifier_net');
pred=sim(net,test_data);
