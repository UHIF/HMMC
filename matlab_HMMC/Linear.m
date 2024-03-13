function [Y] = Linear( Beta,X )
a=-Beta(1)/Beta(2);
b=-Beta(3)/Beta(2);
Y=a.*X+b;
end

