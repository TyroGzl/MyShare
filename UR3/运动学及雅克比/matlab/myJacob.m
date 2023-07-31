function J = myJacob(q)
syms q1 q2 q3 q4 q5 q6
sym_q = [q1 q2 q3 q4 q5 q6];
sym_J = symJacob(sym_q);

J1 = subs(sym_J, q1, q(1));
J2 = subs(J1, q2, q(2));
J3 = subs(J2, q3, q(3));
J4 = subs(J3, q4, q(4));
J5 = subs(J4, q5, q(5));
J6 = subs(J5, q6, q(6));

J = vpa(J6, 8);

end

function [SJ] = symJacob(q)
% syms a2 a3 d1 d2 d4 d5 d6
% UR3»úÐµ±Û²ÎÊý
a2 = -243.65; a3 = -213;
d1 = 151.9; d2 = 119.85; d4 = -9.45; d5 = 83.4; d6 = 82.4;
T1 = [cos(q(1)) -sin(q(1)) 0 0; sin(q(1)) cos(q(1)) 0 0; 0 0 1 d1; 0 0 0 1];
T2 = [cos(q(2)) -sin(q(2)) 0 0; 0 0 -1 -d2; sin(q(2)) cos(q(2)) 0 0; 0 0 0 1];
T3 = [cos(q(3)) -sin(q(3)) 0 a2; sin(q(3)) cos(q(3)) 0 0; 0 0 1 0; 0 0 0 1];
T4 = [cos(q(4)) -sin(q(4)) 0 a3; sin(q(4)) cos(q(4)) 0 0; 0 0 1 d4; 0 0 0 1];
T5 = [cos(q(5)) -sin(q(5)) 0 0; 0 0 -1 -d5; sin(q(5)) cos(q(5)) 0 0; 0 0 0 1];
T6 = [cos(q(6)) -sin(q(6)) 0 0; 0 0 1 d6; -sin(q(6)) -cos(q(6)) 0 0; 0 0 0 1];

T01 = T1;
T02 = T1 * T2;
T03 = T1 * T2 * T3;
T04 = T1 * T2 * T3 * T4;
T05 = T1 * T2 * T3 * T4 * T5;
T06 = T1 * T2 * T3 * T4 * T5 * T6;

px = T06(1, 4); py = T06(2, 4); pz = T06(3, 4);
w1 = T01(1:3, 3); w2 = T02(1:3, 3); w3 = T03(1:3, 3); w4 = T04(1:3, 3); w5 = T05(1:3, 3); w6 = T06(1:3, 3);

J11 = diff(px, q(1)); J12 = diff(px, q(2)); J13 = diff(px, q(3)); J14 = diff(px, q(4)); J15 = diff(px, q(5)); J16 = diff(px, q(6));
J21 = diff(py, q(1)); J22 = diff(py, q(2)); J23 = diff(py, q(3)); J24 = diff(py, q(4)); J25 = diff(py, q(5)); J26 = diff(py, q(6));
J31 = diff(pz, q(1)); J32 = diff(pz, q(2)); J33 = diff(pz, q(3)); J34 = diff(pz, q(4)); J35 = diff(pz, q(5)); J36 = diff(pz, q(6));

SJ = [J11, J12, J13, J14, J15, J16;
    J21, J22, J23, J24, J25, J26;
    J31, J32, J33, J34, J35, J36;
    w1, w2, w3, w4, w5, w6];
end

