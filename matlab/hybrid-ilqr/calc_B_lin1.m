function out1 = calc_B_lin1(in1,in2,in3)
%CALC_B_LIN1
%    OUT1 = CALC_B_LIN1(IN1,IN2,IN3)

%    This function was generated by the Symbolic Math Toolbox version 8.6.
%    12-Nov-2021 22:00:24

m = in3(:,1);
t2 = 1.0./m;
out1 = reshape([0.0,0.0,t2,0.0,0.0,0.0,0.0,t2],[4,2]);