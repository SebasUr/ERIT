%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: vortex + Legendre                                                     %                                                                 
%                                                                              %
% Description: Vortex Legendre Implementation as a function                    %
% Authors: Raul Castaneda                                                      %
% EAFIT                                                                        %
% Medellín, Colombia.                                                          %   
%                                                                              %
% Email: racastaneq@eafit.edu.co                                               %
% version 1.0 (2023)                                                           %

%% clear memory 
clc% clean
close all% close all windows
clear all% clear of memory all variable


%% add folders
% IMPORTANT
inputFolder = 'C:\Users\racastaneq\Documents\MEGA\MEGAsync\RACQ\Universities\05 EAFIT\Research projects\2026\Manon\Samples\holo_1';
outputFolder = 'C:\Users\racastaneq\Documents\MEGA\MEGAsync\RACQ\Universities\05 EAFIT\Research projects\2026\Manon\Samples\holo_1\complex_field_video_1';

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

%% Image extensions to read
imageFiles = [
    dir(fullfile(inputFolder, '*.jpg'));
    dir(fullfile(inputFolder, '*.jpeg'));
    dir(fullfile(inputFolder, '*.png'));
    dir(fullfile(inputFolder, '*.tif'));
    dir(fullfile(inputFolder, '*.tiff'));
    dir(fullfile(inputFolder, '*.bmp'))
];

%% Sort files by name to preserve frame order
fileNames = {imageFiles.name};
[~, sortIdx] = sort_nat(fileNames);
imageFiles = imageFiles(sortIdx);

%% Reconstruction parameters
wavelength = 0.632;
pixelSize = 3.75;
parameter3 = 5;
showFigures = false;
saveFigures = false;
filterRadius = 256/2;

%% Process all holograms
numFiles = length(imageFiles);

fprintf('Number of holograms found: %d\n', numFiles);

for k = 1:numFiles

    currentName = imageFiles(k).name;
    currentPath = fullfile(inputFolder, currentName);

    fprintf('Processing %d / %d: %s\n', k, numFiles, currentName);

    % Read hologram
    [hologram, M, N, m, n] = functions_vortexLegendre.holo_read(currentPath);

    % Apply Vortex + Legendre
    output = vortexLegendre( ...
        hologram, ...
        wavelength, ...
        pixelSize, ...
        parameter3, ...
        true, ...
        true, ...
        filterRadius ...
    );

    % Save complex field in ordered format
    [~, baseName, ~] = fileparts(currentName);

    outputName = sprintf('complex_field_%06d_%s.mat', k, baseName);
    outputPath = fullfile(outputFolder, outputName);

    save(outputPath, 'output', 'currentName', 'currentPath', ...
        'wavelength', 'pixelSize', 'parameter3', 'filterRadius', ...
        'M', 'N', 'm', 'n');

    % Save phase image next to the complex field
    phaseImage = angle(output);
    
    % Normalize phase to [0, 1] only for visualization/saving as image
    phaseImageNorm = mat2gray(phaseImage);
    
    phaseName = sprintf('phase_%06d_%s.png', k, baseName);
    phasePath = fullfile(outputFolder, phaseName);
    
    imwrite(phaseImageNorm, phasePath);

end

fprintf('Batch reconstruction completed.\n');
fprintf('Complex fields saved in:\n%s\n', outputFolder);


function [sortedNames, index] = sort_nat(names)

    if isempty(names)
        sortedNames = {};
        index = [];
        return;
    end

    numbers = zeros(size(names));

    for i = 1:length(names)
        tokens = regexp(names{i}, '\d+', 'match');

        if ~isempty(tokens)
            numbers(i) = str2double(tokens{end});
        else
            numbers(i) = i;
        end
    end

    [~, index] = sort(numbers);
    sortedNames = names(index);

end