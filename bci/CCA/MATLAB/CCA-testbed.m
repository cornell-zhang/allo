% Copyright Allo authors. All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

clear all;
clc;
close all;
% Data path
data_path = 'C:\MATLAB\Matlab Files\BCI\Tsinghua BCI\Subjects\S2.mat';

% Read data
data = importdata(data_path);

% Parameter setting
fs = 250;
len_gaze_s = 4;
len_delay_s = 0.14;
num_harms = 5;
list_freqs = [8:1:15 8.2:1:15.2 8.4:1:15.4 8.6:1:15.6 8.8:1:15.8];
chan_idx = [48,54,55,56,57,58,61,62,63];

% Data preprocessing
eeg = permute(data, [3,1,2,4]);
segment_data = round(len_delay_s*fs)+1 : round(len_delay_s*fs)+round(len_gaze_s*fs);
eeg = eeg(:, chan_idx, segment_data, :);

% Take the data of the first target, first block
eeg_slice = squeeze(eeg(1, :, :, 1))';

% Generate reference signal
freq = list_freqs(10);
t = (1:size(eeg_slice,1))/fs;
ref_signal = [];
for h = 1:num_harms
    ref_signal = [ref_signal, sin(2*pi*h*freq*t)', cos(2*pi*h*freq*t)'];
end
ref_signal = ref_signal(:, 1:2*num_harms);

% Time and execute CCA
tic;
[~,~,r] = canoncorr(eeg_slice, ref_signal);
elapsed_time = toc;

% Output results
fprintf('Maximum correlation coefficient of CCA: %.4f\n', r(1));
fprintf('Running time of canoncorr: %.6f seconds\n', elapsed_time);