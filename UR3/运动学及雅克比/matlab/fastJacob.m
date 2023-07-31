function J = fastJacob(q)
% UR3 机械臂参数
a2 = -243.65; a3 = -213;
d1 = 151.9; d2 = 119.85; d4 = -9.45; d5 = 83.4; d6 = 82.4;

J11 = d6 * (cos(q(1)) * cos(q(5)) + cos(q(2) + q(3) + q(4)) * sin(q(1)) * sin(q(5))) + d2 * cos(q(1)) + d4 * cos(q(1)) - a3 * cos(q(2) + q(3)) * sin(q(1)) - a2 * cos(q(2)) * sin(q(1)) - d5 * sin(q(2) + q(3) + q(4)) * sin(q(1));
J21 = d6 * (cos(q(5)) * sin(q(1)) - cos(q(2) + q(3) + q(4)) * cos(q(1)) * sin(q(5))) + d2 * sin(q(1)) + d4 * sin(q(1)) + a3 * cos(q(2) + q(3)) * cos(q(1)) + a2 * cos(q(1)) * cos(q(2)) + d5 * sin(q(2) + q(3) + q(4)) * cos(q(1));
J31 = 0;
J41 = 0;
J51 = 0;
J61 = 1;

J12 = -cos(q(1)) * (a3 * sin(q(2) + q(3)) + a2 * sin(q(2)) - d5 * cos(q(2) + q(3) + q(4)) - d6 * sin(q(2) + q(3) + q(4)) * sin(q(5)));
J22 = -sin(q(1)) * (a3 * sin(q(2) + q(3)) + a2 * sin(q(2)) - d5 * cos(q(2) + q(3) + q(4)) - d6 * sin(q(2) + q(3) + q(4)) * sin(q(5)));
J32 = a3 * cos(q(2) + q(3)) + a2 * cos(q(2)) + d5 * (cos(q(2) + q(3)) * sin(q(4)) + sin(q(2) + q(3)) * cos(q(4))) - d6 * sin(q(5)) * (cos(q(2) + q(3)) * cos(q(4)) - sin(q(2) + q(3)) * sin(q(4)));
J42 = sin(q(1));
J52 = -cos(q(1));
J62 = 0;

J13 = cos(q(1)) * (d5 * cos(q(2) + q(3) + q(4)) - a3 * sin(q(2) + q(3)) + d6 * sin(q(2) + q(3) + q(4)) * sin(q(5)));
J23 = sin(q(1)) * (d5 * cos(q(2) + q(3) + q(4)) - a3 * sin(q(2) + q(3)) + d6 * sin(q(2) + q(3) + q(4)) * sin(q(5)));
J33 = a3 * cos(q(2) + q(3)) + d5 * sin(q(2) + q(3) + q(4)) - d6 * cos(q(2) + q(3) + q(4)) * sin(q(5));
J43 = sin(q(1));
J53 = -cos(q(1));
J63 = 0;

J14 = cos(q(1)) * (d5 * cos(q(2) + q(3) + q(4)) +d6 * sin(q(2) + q(3) + q(4)) * sin(q(5)));
J24 = sin(q(1)) * (d5 * cos(q(2) + q(3) + q(4)) +d6 * sin(q(2) + q(3) + q(4)) * sin(q(5)));
J34 = d5 * sin(q(2) + q(3) + q(4)) - d6 * cos(q(2) + q(3) + q(4)) * sin(q(5));
J44 = sin(q(1));
J54 = -cos(q(1));
J64 = 0;

J15 = -d6 * (sin(q(1)) * sin(q(5)) + cos(q(2) +q(3) + q(4)) * cos(q(1)) * cos(q(5)));
J25 = d6 * (cos(q(1)) * sin(q(5)) - cos(q(2) +q(3) + q(4)) * cos(q(5)) * sin(q(1)));
J35 = -d6 * sin(q(2) + q(3) + q(4)) * cos(q(5));
J45 = sin(q(2) + q(3) + q(4)) * cos(q(1));
J55 = sin(q(2) + q(3) + q(4)) * sin(q(1));
J65 = -cos(q(2) + q(3) + q(4));

J16 = 0;
J26 = 0;
J36 = 0;
J46 = cos(q(5)) * sin(q(1)) - cos(q(2) +q(3) + q(4)) * cos(q(1)) * sin(q(5));
J56 =- cos(q(1)) * cos(q(5)) - cos(q(2) +q(3) + q(4)) * sin(q(1)) * sin(q(5));
J66 = -sin(q(2) + q(3) + q(4)) * sin(q(5));

J = [J11, J12, J13, J14, J15, J16;
    J21, J22, J23, J24, J25, J26;
    J31, J32, J33, J34, J35, J36;
    J41, J42, J43, J44, J45, J46;
    J51, J52, J53, J54, J55, J56;
    J61, J62, J63, J64, J65, J66
    ];
eps = 1e-5;

for i = 1:6
    
    for j = 1:6
        
        if abs(J(i, j)) < eps
            J(i, j) = 0;
        end
        
    end
    
end

end