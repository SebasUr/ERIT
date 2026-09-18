%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title:-->        parasitemia_main                                                %                                              
%                                                                                  %
% Description:-->  This algorithm allows to evaluate the phase profile of RBCs     %
%                                                                                  %
% Authors:-->      Raul Castaneda                                                  %
%                  EAFIT university                                                %
%                  (Applied Optics Group)                                          %
%                                                                                  %
% Email:-->        racastaneq@eafit.edu.co                                         %
% Date:-->         08/29/2024                                                      %
% version 1.0 (2024)                                                               %
% Notes-->                                                                         %

%% clear memory, workspace, close all figures and clear command window (memory variables)
clc % clean
close all % close all windows
clear all % clear all variables from memory

%% Lines to add folders for reading images and/or functions
% EAFIT computer
imagesRead = dir('C:\Users\USER\Desktop\Anemia\Majo\*.bmp');
dir = 'C:\Users\USER\Desktop\Anemia\Majo'; 
folder_phase = 'C:\Users\USER\Desktop\Anemia\Majo_Resultados';

%% Lines to load parameters and define constants
M = 1280;
N = 960;
lambda = 0.633;
dxy = 3.75;
k = 2 * pi / lambda;
fx_0 = M / 2;
fy_0 = N / 2;

%% Load the hologram
for cont = 1:length(imagesRead)
    name = imagesRead(cont).name; 
     fprintf('El nombre del holograma es: %s El numero %.2f\n', name, cont);
     indice = find(strcmp({imagesRead.name}, name));
    filename = fullfile(dir, name);
    [hologram, M, N, m, n] = functions_anemia.holo_read(filename);

    % Lines to implement the spatial filter using a circular mask
    [holo_filtered, fx_max, fy_max] = functions_anemia.spatial_filter(hologram, M, N, 'Not', 3);

    % minimization fminsearch
    seed_maxPeak = [fx_max - 1, fy_max - 1];
    options = optimset('Display', 'off', 'MaxIter', 100, 'TolX', 1e-6);
    [MaxPeaks, cf_fminsearch] = fminsearch(@(t)(functions_anemia.cost_function(t, lambda, dxy, M, N, m, n, k, fx_0, fy_0, holo_filtered)), seed_maxPeak, options);

    fx_max_temp = MaxPeaks(1, 1);
    fy_max_temp = MaxPeaks(1, 2);

    fprintf('fx: %f\n', fx_max_temp);
    fprintf('fy: %f\n', fy_max_temp);
 
    % Best reconstruction
    ref_wave = functions_anemia.reference_wave(M, N, m, n, lambda, dxy, fx_max_temp, fy_max_temp, k, fx_0, fy_0);
    field_compensate = ref_wave .* holo_filtered;
    phase = angle(field_compensate);
    figure, imagesc(phase), colormap(gray), title('Phase fminsearch_notP'), daspect([1 1 1]);

    % Profile Zones 
    numZones = input('Enter the number of zones to analyze (0 to skip): ');
    
    % Check if numZones is 0 to skip to the next image
    if numZones == 0
        disp('Skipping to the next image...');
        continue; % Skip to the next iteration of the for loop
    end
    
    % [diffeAverage] = functions_anemia.profiles(phase, numZones, 'True');
    [diffeAverage] = functions_anemia.circulos(phase, numZones);

    % Save phase images
    name = erase(imagesRead(cont).name, ".bmp");
    phase_save = uint8(255 * mat2gray(phase));
    folder_p = fullfile(folder_phase, strcat(name, "_phase", num2str(round(diffeAverage, 2)), '.bmp'));
    imwrite(phase_save, folder_p);
end
