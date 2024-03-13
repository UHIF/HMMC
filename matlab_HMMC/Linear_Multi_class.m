%%对所有的每次不成立的都增加其比例。没有对权值进行限制
clc;
clear;
load('HMMC_multi_class')

% figure
% plot(X(1:34,1),X(1:34,2),'r.','MarkerSize',15); hold on;
% plot(X(35:45,1),X(35:45,2),'b.','MarkerSize',15); hold on;
% plot(X(46:70,1),X(46:70,2),'g.','MarkerSize',15); 
rec_dis_w=[];
min_rec=[];
p_rec_1=[];
p_rec_2=[];
p_rec_3=[];
min_rec_1=[];
min_rec_2=[];
min_rec_3=[];

r_p=0.01;

n1=34;
n2=11;
n3=25;


p1=0;%%类别1和2
p2=0;%%类别1和3
p3=0;%%类别2和3

r=1.01;
iteration=1000;
[n,nm]=size(X);
bias(1:n)=1;
bias=bias';
X=cat(2,X,bias);

% [m,~]=size(X2);
% bias2(1:m)=1;
% bias2=bias2';
% X2=cat(2,X2,bias2);
% 加权最小二乘
W=zeros(n);
for i=1:n
    W(i,i)=1;
end
Test=[];
Train=[];
rl=1;
for l=1:iteration
Beta = (X'*W*W*X+eye(nm+1)*rl)\X'*W*W*T;
dis_w=0;
for i=1:nm
dis_w=dis_w+sqrt(Beta(1:end-1,i)'*Beta(1:end-1,i));
end
X_test=X*Beta;
X1=X_test(1:n1,:);
X2=X_test(n1+1:n1+n2,:);
X3=X_test(n1+n2+1:n1+n2+n3,:);
%%
%%类别1和2
A12=find(X1(:,1)-X1(:,2)<p1);
A21=n1+find(X2(:,2)-X2(:,1)<p1);
if isempty(A12) && isempty(A21)
      p1=p1+r_p;
      continue;
else
    min1=min(min(X1(:,1)-X1(:,2)),min(X2(:,2)-X2(:,1)));
end

%%
%%类别1和3
A13=find(X1(:,1)-X1(:,3)<p2);
A31=n1+n2+find(X3(:,3)-X3(:,1)<p2);

if isempty(A13)&&isempty(A31)
      p2=p2+r_p;
      continue;
else
    min2=min(min(X1(:,1)-X1(:,3)),min(X3(:,3)-X3(:,1)));
end
%%
%%类别2和3
A23=n1+find(X2(:,2)-X2(:,3)<p3);


A32=n1+n2+find(X3(:,3)-X3(:,2)<p3);

if isempty(A23) && isempty(A32)
      p3=p3+r_p;
      continue;
else
    min3=min(min(X2(:,2)-X2(:,3)),min(X3(:,3)-X3(:,2))); 
end
%%
A=[A12;A21;A13;A31;A23;A32];

O=Tools.OneHotMatrix(X_test);
Rate=1-confusion(O',T');
Train=cat(1,Train,Rate);

m=size(A);
if p1>=1 && p2>=1 && p3>=1
    break;
end

if isempty(A)
      p=p+r_p;
      continue;
end


for j=1:m
    W(A(j),A(j))=W(A(j),A(j))*r;
end
rec_dis_w=cat(2,rec_dis_w,dis_w);

min_rec_1=cat(2,min_rec_1,min1);
min_rec_2=cat(2,min_rec_2,min2);
min_rec_3=cat(2,min_rec_3,min3);


p_rec_1=cat(2,p_rec_1,p1);
p_rec_2=cat(2,p_rec_2,p2);
p_rec_3=cat(2,p_rec_3,p3);


% min_rec=cat(2,min_rec,minA);
% W=n*W/sum(diag(W));

% X_test=X2*Beta;
% O2=Tools.OneHotMatrix(X_test);
% Rate=1-confusion(O2',T2')
% Test=cat(1,Test,Rate);

end
figure
plot(rec_dis_w)

figure
plot(p_rec_2,'LineWidth',1.5);hold on;
plot(min_rec_2,'LineWidth',1.5);hold on;
plot(p_rec_1,'LineWidth',1.5);hold on;
plot(min_rec_1,'LineWidth',1.5);hold on;
plot(p_rec_3,'LineWidth',1.5);hold on;
plot(min_rec_3,'LineWidth',1.5);hold on;

legend('Threshold1 ','Margin1','Threshold2 ','Margin2','Threshold3 ','Margin3')
xlabel('Iteration');



% 
% figure
% plot(300:340,p_rec_2(300:340),'LineWidth',1.5);hold on;
% plot(300:340,min_rec_2(300:340),'LineWidth',1.5);hold on;
% 
% figure
% plot(550:600,p_rec_1(550:600),'LineWidth',1.5);hold on;
% plot(550:600,min_rec_1(550:600),'LineWidth',1.5);hold on;
% 
% figure
% plot(450:500,p_rec_3(450:500),'LineWidth',1.5);hold on;
% plot(450:500,min_rec_3(450:500),'LineWidth',1.5);hold on;

% %%最小二乘
% Beta=pinv(X)*T;
% X_train=X*Beta;
% O=Tools.OneHotMatrix(X_train);
% Rate=1-confusion(O',T')

% X_test=X2*Beta;
% O2=Tools.OneHotMatrix(X_test);
% Rate_test=1-confusion(O2',T2')
