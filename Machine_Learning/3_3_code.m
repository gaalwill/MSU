% ===== Generate signal =====
n = 1024;                       % resolution
t = linspace(0, 0.15, n);       % 0 to 0.15 seconds
T = t(end) - t(1);

x = cos(2*pi*97*t) + cos(2*pi*777*t);	%Generated Signal

xt = fft(x);
PSD = abs(xt).^2 / n;
f = (0:n-1) / T;                % frequency axis in Hz


%% ===== Random sampling =====
p = 20;                         % number of measurements
perm = randperm(n, p);
y = x(perm);


%% ===== Compressed sensing =====
Psi = dct(eye(n));
Theta = Psi(perm, :);

s = CoSaMP(y', Theta, 10, 1e-10);
xrecon = idct(s);

%% ---- Figure 1: Original signal (time domain) ----
figure;
plot(t, x, 'k', 'LineWidth', 1);
xlim([0 0.02])
ylim([-2.25 2.25])
xlabel('Time (s)');
ylabel('Amplitude');
title('Original Signal');
grid on;


%% ---- Figure 2: Original signal PSD ----
figure;
plot(f, PSD, 'LineWidth', 1);
xlim([0 1000]);
xlabel('Frequency (Hz)');
ylabel('Power');
title('Original Signal Power Spectrum');
grid on;

%% ---- Figure 3: Reconstructed signal (time domain) ----
figure;
plot(t, real(xrecon), 'r', 'LineWidth', 1);
xlim([0 0.02])
ylim([-2.25 2.25])
xlabel('Time (s)');
ylabel('Amplitude');
title('Reconstructed Signal');
grid on;


%% ---- Figure 4: Reconstructed signal PSD ----
figure;
plot(f, abs(fft(xrecon)).^2 / n, 'LineWidth', 1);
xlim([0 1000]);
xlabel('Frequency (Hz)');
ylabel('Power');
title('Reconstructed Signal Power Spectrum');
grid on;
