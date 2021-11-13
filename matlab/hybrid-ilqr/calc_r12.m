function R1 = calc_r12(in1,in2,in3)
%CALC_R12
%    R1 = CALC_R12(IN1,IN2,IN3)

%    This function was generated by the Symbolic Math Toolbox version 8.6.
%    12-Nov-2021 22:19:16

x = in1(1,:);
x_dot = in1(3,:);
y = in1(2,:);
y_dot = in1(4,:);
t2 = x.*y_dot;
t3 = x_dot.*y;
t4 = x.^2;
t5 = x_dot.*5.0;
t6 = y.^2;
t7 = y.*1.0e+1;
t8 = -t3;
t9 = -t7;
t10 = t2+t5+t8;
t11 = t4+t6+t9+2.5e+1;
t12 = 1.0./t11;
R1 = [x;y;-t10.*t12.*(y-5.0);t10.*t12.*x];
