function A_lin1_disc = calc_A_lin1_disc(in1,in2,dt,in4)
%CALC_A_LIN1_DISC
%    A_LIN1_DISC = CALC_A_LIN1_DISC(IN1,IN2,DT,IN4)

%    This function was generated by the Symbolic Math Toolbox version 8.6.
%    12-Nov-2021 22:00:25

A_lin1_disc = reshape([1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,dt,0.0,1.0,0.0,0.0,dt,0.0,1.0],[4,4]);