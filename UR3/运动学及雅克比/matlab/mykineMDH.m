eps = 1e-5; %С��������ʹ��
P = 180 / pi; %�Ƕ��ƺͻ�����ת������
% UR3��е�۲���
a2 = -243.65; a3 = -213;
d1 = 151.9; d2 = 119.85; d4 = -9.45; d5 = 83.4; d6 = 82.4;
%ѡ�����˶�ѧ�����˶�ѧ
flag = input('��ѡ���˶�ѧ��ʽ��\n1�����˶�ѧ��\n2�����˶�ѧ\n');
%**************************���˶�ѧ����****************************%
if flag == 1
    q = input('������ؽڽǣ������ƣ���\n');
%     q = [15 15 15 15 15 15] / P;
    T1 = [cos(q(1)) -sin(q(1)) 0 0; sin(q(1)) cos(q(1)) 0 0; 0 0 1 d1; 0 0 0 1];
    T2 = [cos(q(2)) -sin(q(2)) 0 0; 0 0 -1 -d2; sin(q(2)) cos(q(2)) 0 0; 0 0 0 1];
    T3 = [cos(q(3)) -sin(q(3)) 0 a2; sin(q(3)) cos(q(3)) 0 0; 0 0 1 0; 0 0 0 1];
    T4 = [cos(q(4)) -sin(q(4)) 0 a3; sin(q(4)) cos(q(4)) 0 0; 0 0 1 d4; 0 0 0 1];
    T5 = [cos(q(5)) -sin(q(5)) 0 0; 0 0 -1 -d5; sin(q(5)) cos(q(5)) 0 0; 0 0 0 1];
    T6 = [cos(q(6)) -sin(q(6)) 0 0; 0 0 1 d6; -sin(q(6)) -cos(q(6)) 0 0; 0 0 0 1];
    FT = T1 * T2 * T3 * T4 * T5 * T6
end

%*************************���˶�ѧ����*****************************%
if flag == 2
    fflag = input('��ѡ���ʾ��ʽ��\n1��λ�˾���\n2����תʸ��\n')

    if fflag == 1
        FT = input('������λ�˾���\n');
        nx = FT(1, 1); ny = FT(2, 1); nz = FT(3, 1);
        ox = FT(1, 2); oy = FT(2, 2); oz = FT(3, 2);
        ax = FT(1, 3); ay = FT(2, 3); az = FT(3, 3);
        px = FT(1, 4); py = FT(2, 4); pz = FT(3, 4);
        T = [nx ox ax px;
            ny oy ay py;
            nz oz az pz;
            0 0 0 1];
        m = py - ay * d6;
        n = px - ax * d6;
    elseif fflag == 2
        pose = input('��������תʸ����\n');
        px = pose(1);
        py = pose(2);
        pz = pose(3);
        r1 = pose(4);
        r2 = pose(5);
        r3 = pose(6);
        theta = sqrt(r1^2 + r2^2 + r3^2);
        kx = r1 / theta;
        ky = r2 / theta;
        kz = r3 / theta;
        nx = kx * kx * (1 - cos(theta)) + cos(theta);

        ox = kx * ky * (1 - cos(theta)) - kz * sin(theta);
        ax = kx * kz * (1 - cos(theta)) + ky * sin(theta);
        ny = kx * ky * (1 - cos(theta)) + kz * sin(theta);
        oy = ky * ky * (1 - cos(theta)) + cos(theta);
        ay = ky * kz * (1 - cos(theta)) - kx * sin(theta);
        nz = kx * kz * (1 - cos(theta)) - ky * sin(theta);
        oz = ky * kz * (1 - cos(theta)) + kx * sin(theta);
        az = kz * kz * (1 - cos(theta)) + cos(theta);
    end

    last_theta_6 = input('�������ϴιؽ�6�ĽǶȣ�\n');
    % last_theta_6 = q(6);

    count = 1; %������

    for index_i = [1, 2]
        theta_1 = atan2(m, n) - atan2(-d2 - d4, (-1)^index_i * sqrt(m^2 + n^2 - (d2 + d4)^2));

        if abs(theta_1) < eps
            theta_1 = 0;
        end

        for index_j = [1, 2]
            c5 = ax * sin(theta_1) - ay * cos(theta_1);

            if abs(c5) > 1
                disp(['��', num2str(count), '���Ϊ��'])
                disp('�޽�')
                count = count + 1;
                continue
            end

            theta_5 = (-1)^index_j * acos(c5);

            if abs(theta_5) < eps
                theta_5 = 0;
            end

            s = cos(theta_1) * ny - sin(theta_1) * nx;
            t = cos(theta_1) * oy - sin(theta_1) * ox;

            if theta_5 == 0
                theta_6 = last_theta_6; %���theta_5Ϊ�㣬���죬ȡtheta_6Ϊ�ϴη���ĽǶȡ�
            else
                theta_6 = atan2(s, t) - atan2(-sin(theta_5), 0);
            end

            if abs(theta_6) < eps
                theta_6 = 0;
            end

            r14 = px * cos(theta_1) + sin(theta_1) * py - d6 * (sin(theta_1) * ay + cos(theta_1) * ax) + d5 * (cos(theta_1) * cos(theta_6) * ox + cos(theta_1) * sin(theta_6) * nx + cos(theta_6) * sin(theta_1) * oy + sin(theta_1) * sin(theta_6) * ny);
            r34 = pz - d1 - az * d6 + d5 * (sin(theta_6) * nz + cos(theta_6) * oz);

            for index_k = [1, 2]
                c3 = ((r14)^2 + (r34)^2 - a3^2 - a2^2) / (2 * a2 * a3);

                if abs(c3) > 1
                    disp(['��', num2str(count), '���Ϊ��'])
                    disp('�޽�')
                    count = count + 1;
                    continue
                end

                theta_3 = (-1)^index_k * acos(c3);

                if abs(theta_3) < eps
                    theta_3 = 0;
                end

                s2 = (r34 * (a3 * cos(theta_3) + a2) - a3 * sin(theta_3) * r14) / (a3^2 + a2^2 + 2 * a2 * a3 * cos(theta_3));
                c2 = (r34 * a3 * sin(theta_3) + (a3 * cos(theta_3) + a2) * r14) / (a3^2 + a2^2 + 2 * a2 * a3 * cos(theta_3));
                theta_2 = atan2(s2, c2);

                if abs(theta_2) < eps
                    theta_2 = 0;
                end

                s234 = -sin(theta_6) * (cos(theta_1) * nx + sin(theta_1) * ny) - cos(theta_6) * (cos(theta_1) * ox + sin(theta_1) * oy);
                c234 = sin(theta_6) * nz + cos(theta_6) * oz;
                theta_4 = atan2(s234, c234) - theta_2 - theta_3;

                if abs(theta_4) < eps
                    theta_4 = 0;
                end

                % ���
                disp(['��', num2str(count), '���Ϊ��']);
                count = count + 1;
                %�Ƕ����
                disp(['theta = [', num2str((theta_1 * P)), ' ', num2str((theta_2 * P)), ' ', num2str((theta_3 * P)), ' ', num2str((theta_4 * P)), ' ', num2str((theta_5 * P)), ' ', num2str((theta_6 * P)), ']'])
                %�������
%                 disp(['theta = [', num2str((theta_1)), ' ', num2str((theta_2)), ' ', num2str((theta_3)), ' ', num2str((theta_4)), ' ', num2str((theta_5)), ' ', num2str((theta_6)), ']'])
            end

        end

    end

end
