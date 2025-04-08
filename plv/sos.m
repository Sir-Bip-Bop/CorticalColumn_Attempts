sum_control = 0;
P = '.'; 
Control = dir(fullfile(P,'C*.mat'));
for k = 1:numel(Control)
F = fullfile(P,Control(k).name);
C = load(F);
Control(k).data = C;
sum_control = sum_control + Control(k).data.band(2).plv_rms
end

sum_control = sum_control/7

sum_high = 0;
P = '.'; 
High = dir(fullfile(P,'H*.mat'));
for k = 1:numel(High)
F = fullfile(P,High(k).name);
C = load(F);
High(k).data = C;
sum_high = sum_high + High(k).data.band(2).plv_rms
end
sum_high = sum_high/10

sum_excelent = 0;
P = '.'; 
Excel = dir(fullfile(P,'E*.mat'));
for k = 1:numel(Excel)
F = fullfile(P,Excel(k).name);
C = load(F);
Excel(k).data = C;
sum_excelent = sum_excelent + Excel(k).data.band(2).plv_rms
end
sum_excelent = sum_excelent/2

figure(1)
surf(sum_control)
colorbar

figure(2)
surf(sum_high)
colorbar

figure(3)
surf(sum_excelent)
colorbar


figure(4)
surf(sum_high-sum_control)
colorbar


figure(5)
surf(sum_excelent-sum_control)
colorbar