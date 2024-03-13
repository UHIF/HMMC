clc;
clear;
load('HMMC_binary')

p_k=[];
p=0; 
r=1.01;  
iteration=2000;
[n,nm]=size(X);
W=zeros(n);
for i=1:n
    W(i,i)=1;
end
bias=ones(n,1);
dis_k=[];
rec_dis_w=[];
rec_v=[];
rec_min=[];
X2=cat(2,X,bias);
for l=1:iteration
     Beta = (X2'*W*X2)\X2'*W*T;
     dis_w=sqrt(Beta(1:end-1)'*Beta(1:end-1));
     X_test=X2*Beta;
     O=X_test.*T;
     A=find(O<p);
     [m,~]=size(A);
     Accuracy=(n-m)/n;
if isempty(A) 
     p=p+0.01;
 end
dis=min(O)/dis_w;
rec_min=cat(2,rec_min,min(O));
rec_dis_w=cat(2,rec_dis_w,dis_w);
dis_k=cat(2,dis_k,dis);
for j=1:m
    W(A(m),A(m))=W(A(m),A(m))*r;
end
end
plot(rec_dis_w);
figure;
plot(dis_k);
figure;
hold on;
plot(rec_min);
ylim([0,1]);
figure
plot(X(1:10,1),X(1:10,2),'o','MarkerSize',8,'MarkerFaceColor','r','MarkerEdgeColor','k'); hold on;
plot(X(11:20,1),X(11:20,2),'o','MarkerSize',8,'MarkerFaceColor','b','MarkerEdgeColor','k'); hold on;
legend('Positive samples','Negative samples');

X_linear=[0,4];
Y_linear=Linear(Beta,X_linear);
plot(X_linear,Y_linear,'g-','LineWidth',2); hold on;
legend('Positive samples','Negative samples','Linear programming');
xlim([0,4]);
ylim([0,4]);
