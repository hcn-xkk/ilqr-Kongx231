function out1 = calc_B_lin2(in1,in2,in3)
%CALC_B_LIN2
%    OUT1 = CALC_B_LIN2(IN1,IN2,IN3)

%    This function was generated by the Symbolic Math Toolbox version 8.6.
%    12-Nov-2021 22:00:24

m = in3(:,1);
x = in1(1,:);
y = in1(2,:);
t2 = x.^2;
t3 = y.^2;
t4 = m.*2.5e+1;
t5 = y-5.0;
t6 = m.*y.*1.0e+1;
t7 = m.*t2;
t8 = m.*t3;
t9 = -t6;
t10 = t4+t7+t8+t9;
t11 = 1.0./t10;
t12 = t5.*t11.*x;
t13 = -t12;
out1 = reshape([0.0,0.0,t5.^2.*t11,t13,0.0,0.0,t13,t2.*t11],[4,2]);