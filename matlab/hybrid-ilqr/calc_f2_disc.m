function f2_disc = calc_f2_disc(in1,in2,dt,in4)
%CALC_F2_DISC
%    F2_DISC = CALC_F2_DISC(IN1,IN2,DT,IN4)

%    This function was generated by the Symbolic Math Toolbox version 8.6.
%    12-Nov-2021 22:00:24

g = in4(:,2);
m = in4(:,1);
u_x = in2(1,:);
u_y = in2(2,:);
x = in1(1,:);
x_dot = in1(3,:);
y = in1(2,:);
y_dot = in1(4,:);
t2 = g.*m;
t3 = x.^2;
t4 = x_dot.^2;
t5 = y.^2;
t6 = y_dot.^2;
t7 = m.*2.5e+1;
t8 = y.*1.0e+1;
t9 = y-5.0;
t17 = m.*y.*-1.0e+1;
t10 = -t2;
t11 = t4.*2.0;
t12 = t6.*2.0;
t13 = m.*t8;
t14 = m.*t3;
t15 = m.*t5;
t16 = -t8;
t18 = t10+u_y;
t19 = t11+t12;
t20 = t3+t5+t16+2.5e+1;
t22 = t7+t14+t15+t17;
t21 = 1.0./t20;
t23 = 1.0./t22;
f2_disc = [x+dt.*x_dot;y+dt.*y_dot;x_dot+dt.*(t9.^2.*t23.*u_x-(t19.*t21.*x)./2.0+t9.*t23.*x.*(t2-u_y));y_dot-dt.*((t9.*t19.*t21)./2.0+t3.*t23.*(t2-u_y)+t9.*t23.*u_x.*x)];
