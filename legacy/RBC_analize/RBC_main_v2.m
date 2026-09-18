%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title:           RBC_main
%
% Description:     This algorithm evaluates phase profiles of manually selected
%                  red blood cells using two diagonal profiles per cell.
%
% Authors:         Manon and Raul Castaneda
%                  EAFIT University
%                  Applied Optics Group
%
% Email:           racastaneq@eafit.edu.co
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
close all;
clear;


%% Load complex field

folderPath = ['C:\Users\racastaneq\Documents\MEGA\MEGAsync\RACQ\' ...
    'Universities\05 EAFIT\Research projects\2026\Manon\' ...
    'Samples\holo_1\complex_field_video_1\'];

fileName = 'complex_field_000002_Hologram_video01_frame_000002.mat';

data = load(fullfile(folderPath, fileName));


%% Identify complex-field variable

if isfield(data, 'output')
    field = data.output;
else
    variableNames = fieldnames(data);

    if isempty(variableNames)
        error('The MAT file does not contain any variables.');
    end

    field = data.(variableNames{1});

    warning( ...
        'Variable "output" was not found. Variable "%s" was used.', ...
        variableNames{1});
end


%% Calculate amplitude and phase

amplitude = abs(field);
phase = angle(field);


%% Display phase image

figure('Name', 'Original phase image');
imagesc(phase);
axis image;
colormap(gray);
colorbar;
title('Original phase image [rad]');


%% Normalize and binarize phase image

phaseNormalized = mat2gray(phase);

% Adjustable threshold
binaryThreshold = 0.30;

binaryImage = imbinarize(phaseNormalized, binaryThreshold);

figure('Name', 'Binary phase image');
imshow(binaryImage);
title(sprintf('Binary phase image — Threshold = %.2f', binaryThreshold));


%% Manual image-quality evaluation

qualityAnswer = questdlg( ...
    'Is this phase image suitable for RBC analysis?', ...
    'Image quality evaluation', ...
    'Good', ...
    'Bad', ...
    'Good');

if isempty(qualityAnswer)
    disp('Analysis cancelled by the user.');
    return;
end

if strcmpi(qualityAnswer, 'Bad')
    fprintf('The image was classified as BAD and will not be analyzed.\n');
    return;
end

fprintf('The image was classified as GOOD.\n');


%% Select RBCs and calculate diagonal profiles

numRBCs = input('Enter the number of RBCs to analyze: ');

if isempty(numRBCs) || numRBCs < 1 || mod(numRBCs, 1) ~= 0
    error('The number of RBCs must be a positive integer.');
end

[resultsTable, globalAverage] = ...
    functions_anemia.diagonalRBCProfiles(phase, numRBCs, true);


%% Display results

disp(' ');
disp('Individual RBC measurements:');
disp(resultsTable);

fprintf('\nGlobal average phase difference: %.4f rad\n', globalAverage);


%% Save results

[~, baseName, ~] = fileparts(fileName);

outputTableName = fullfile( ...
    folderPath, ...
    [baseName, '_phase_measurements.csv']);

writetable(resultsTable, outputTableName);

fprintf('Results saved in:\n%s\n', outputTableName);