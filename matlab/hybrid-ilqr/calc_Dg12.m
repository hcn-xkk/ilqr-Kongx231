function DG1 = calc_Dg12(in1,in2,in3)
%CALC_DG12
%    DG1 = CALC_DG12(IN1,IN2,IN3)

%    This function was generated by the Symbolic Math Toolbox version 8.6.
%    12-Nov-2021 22:00:23

x = in1(1,:);
y = in1(2,:);
DG1 = [x.*2.0,y.*2.0-1.0e+1,0.0,0.0];