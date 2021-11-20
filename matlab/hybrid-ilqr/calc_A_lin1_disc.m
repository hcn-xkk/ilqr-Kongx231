function A_lin1_disc = calc_A_lin1_disc(in1,in2,dt,in4)
%CALC_A_LIN1_DISC
%    A_LIN1_DISC = CALC_A_LIN1_DISC(IN1,IN2,DT,IN4)

%    This function was generated by the Symbolic Math Toolbox version 8.6.
%    19-Nov-2021 19:55:09

L1 = in4(:,2);
L2 = in4(:,4);
g = in4(:,5);
m1 = in4(:,1);
m2 = in4(:,3);
q1 = in1(1,:);
q2 = in1(2,:);
q1_dot = in1(3,:);
q2_dot = in1(4,:);
tau_1 = in2(1,:);
tau_2 = in2(2,:);
t2 = cos(q1);
t3 = cos(q2);
t4 = sin(q1);
t5 = sin(q2);
t6 = q1+q2;
t7 = L1.^2;
t8 = L1.^3;
t9 = L2.^2;
t10 = L2.^3;
t11 = m2.^2;
t12 = q1_dot.^2;
t13 = q2_dot.^2;
t17 = 1.0./L2;
t20 = 1.0./m2;
t14 = t3.^2;
t15 = t5.^2;
t16 = cos(t6);
t18 = 1.0./t9;
t19 = sin(t6);
t21 = m1.*t7.*4.0;
t22 = m2.*t7.*1.2e+1;
t23 = L1.*m2.*q1_dot.*t5.*t9.*4.0;
t24 = L1.*m2.*q2_dot.*t5.*t9.*4.0;
t27 = L1.*q1_dot.*t5.*t10.*t11.*4.0;
t28 = L1.*q2_dot.*t5.*t10.*t11.*4.0;
t36 = q2_dot.*t3.*t5.*t7.*t9.*t11.*6.0;
t25 = m2.*t7.*t14.*9.0;
t26 = L1.*L2.*g.*m2.*t3.*t19.*3.0;
t30 = L2.*g.*m1.*m2.*t7.*t19.*2.0;
t32 = L2.*g.*t7.*t11.*t19.*6.0;
t35 = L1.*g.*t3.*t9.*t11.*t19.*3.0;
t29 = -t25;
t31 = -t26;
t33 = -t30;
t34 = -t32;
t37 = -t35;
t38 = t21+t22+t29;
t39 = 1.0./t38;
t40 = t39.^2;
A_lin1_disc = reshape([1.0,0.0,dt.*t17.*t39.*(t31+L1.*L2.*g.*m1.*t4.*2.0+L1.*L2.*g.*m2.*t4.*4.0).*3.0,dt.*t18.*t20.*t39.*(t33+t34+t37+L1.*g.*t4.*t9.*t11.*4.0+L1.*g.*m1.*m2.*t4.*t9.*2.0+L2.*g.*t3.*t4.*t7.*t11.*6.0+L2.*g.*m1.*m2.*t3.*t4.*t7.*3.0).*-3.0,0.0,1.0,dt.*t17.*t39.*(t31+L1.*t5.*tau_2.*6.0+L1.*m2.*t3.*t9.*t12.*2.0+L1.*m2.*t3.*t9.*t13.*2.0+L2.*m2.*t7.*t12.*t14.*3.0-L2.*m2.*t7.*t12.*t15.*3.0-L1.*L2.*g.*m2.*t5.*t16.*3.0+L1.*m2.*q1_dot.*q2_dot.*t3.*t9.*4.0).*3.0-dt.*m2.*t3.*t5.*t7.*t17.*t40.*(L2.*tau_1.*4.0-L2.*tau_2.*4.0+q2_dot.*t23-L1.*t3.*tau_2.*6.0-L1.*L2.*g.*m1.*t2.*2.0-L1.*L2.*g.*m2.*t2.*4.0+L1.*m2.*t5.*t9.*t12.*2.0+L1.*m2.*t5.*t9.*t13.*2.0+L1.*L2.*g.*m2.*t3.*t16.*3.0+L2.*m2.*t3.*t5.*t7.*t12.*3.0).*5.4e+1,dt.*t18.*t20.*t39.*(t33+t34+t37-L1.*L2.*m2.*t5.*tau_1.*6.0+L1.*L2.*m2.*t5.*tau_2.*1.2e+1+L2.*t3.*t8.*t11.*t12.*6.0+L1.*t3.*t10.*t11.*t12.*2.0+L1.*t3.*t10.*t11.*t13.*2.0+t7.*t9.*t11.*t12.*t14.*6.0-t7.*t9.*t11.*t12.*t15.*6.0+t7.*t9.*t11.*t13.*t14.*3.0-t7.*t9.*t11.*t13.*t15.*3.0+q1_dot.*q2_dot.*t7.*t9.*t11.*t14.*6.0-q1_dot.*q2_dot.*t7.*t9.*t11.*t15.*6.0+L2.*m1.*m2.*t3.*t8.*t12.*2.0+L2.*g.*t2.*t5.*t7.*t11.*6.0-L1.*g.*t5.*t9.*t11.*t16.*3.0+L1.*q1_dot.*q2_dot.*t3.*t10.*t11.*4.0+L2.*g.*m1.*m2.*t2.*t5.*t7.*3.0).*-3.0+dt.*t3.*t5.*t7.*t18.*t40.*(q2_dot.*t27+q1_dot.*t36-m1.*t7.*tau_2.*4.0-m2.*t7.*tau_2.*1.2e+1+m2.*t9.*tau_1.*4.0-m2.*t9.*tau_2.*4.0+L1.*L2.*m2.*t3.*tau_1.*6.0-L1.*L2.*m2.*t3.*tau_2.*1.2e+1-L1.*g.*t2.*t9.*t11.*4.0+L2.*g.*t7.*t11.*t16.*6.0+L2.*t5.*t8.*t11.*t12.*6.0+L1.*t5.*t10.*t11.*t12.*2.0+L1.*t5.*t10.*t11.*t13.*2.0+t3.*t5.*t7.*t9.*t11.*t12.*6.0+t3.*t5.*t7.*t9.*t11.*t13.*3.0-L1.*g.*m1.*m2.*t2.*t9.*2.0+L2.*g.*m1.*m2.*t7.*t16.*2.0+L2.*m1.*m2.*t5.*t8.*t12.*2.0-L2.*g.*t2.*t3.*t7.*t11.*6.0+L1.*g.*t3.*t9.*t11.*t16.*3.0-L2.*g.*m1.*m2.*t2.*t3.*t7.*3.0).*5.4e+1,dt,0.0,dt.*t17.*t39.*(t23+t24+L2.*m2.*q1_dot.*t3.*t5.*t7.*6.0).*3.0+1.0,dt.*t18.*t20.*t39.*(t27+t28+t36+L2.*q1_dot.*t5.*t8.*t11.*1.2e+1+q1_dot.*t3.*t5.*t7.*t9.*t11.*1.2e+1+L2.*m1.*m2.*q1_dot.*t5.*t8.*4.0).*-3.0,0.0,dt,dt.*t17.*t39.*(t23+t24).*3.0,dt.*t18.*t20.*t39.*(t27+t28+t36+q1_dot.*t3.*t5.*t7.*t9.*t11.*6.0).*-3.0+1.0],[4,4]);
