%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title:-->        parasitemia_main                                                %                                              
%                                                                                  %
% Description:-->  This algorithm allows to evalaute the phase profile of RBCs     %
%                                                                                  %
% Authors:-->      Raul Castaneda                                                  %
%                  EAFIT univeristy                                                %
%                  (Applied Optics Group)                                          %
%                                                                                  %
% Email:-->        racastaneq@eafit.edu.co                                         %
% Date:-->         08/29/2024                                                      %
% version 1.0 (2024)                                                               %
% Notes-->                                                                         %


%% clear memory, worksapce,close all figures and clear command window (memory variables)
clc% clean
close all% close all windows
clear all% clear of memory all variable


%% load complex field

addpath('C:\Users\racastaneq\Documents\MEGA\MEGAsync\RACQ\Universities\05 EAFIT\Research projects\2026\Manon\Samples\holo_1\complex_field_video_1\')
data = load('complex_field_000002_Hologram_video01_frame_000002.mat');

% Complex field
field = data.output;     

amplitude = abs(field);          
phase = angle(field);

figure,imagesc(phase),colormap(gray),title('Phase fminsearch'),daspect([1 1 1])

%% Profile Zones 
numZones = input('Enter the number of zones to analyze: ');
[table] = functions_anemia.profiles(phase, numZones, 'true');


