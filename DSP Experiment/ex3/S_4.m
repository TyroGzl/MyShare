clear all
b1 = [0 sin(pi / 25)];
a1 = [1 -2 * cos(pi / 25) 1];
b2 = [1 -cos(pi / 25)];
a2 = [1 -2 * cos(pi / 25) 1];

impz(2 * b1, a1, 100);
hold on
impz(2 * b2, a2, 100);
legend('’˝œ“–Ú¡–', '”‡œ“–Ú¡–');
