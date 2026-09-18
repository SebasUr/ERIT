%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: functions_evaluation                                                   %
%                                                                              %
% The script contains all implemented function for the evaluation_main.pyget   %
%                                                                              %                                       
% Authors: Raul Castaneda and Ana Doblas                                       %
% Applied Optics Group EAFIT univeristy                                        % 
%                                                                              %
% Email: racastaneq@eafit.edu.co; adoblas@umassd.edu                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef functions_anemia
    methods(Static)

        function [holo,M,N,m,n] = holo_read(filename)
            holo = double(imread(filename));
            holo = holo(:,:,1);
            [N,M] = size(holo);
            [n,m] = meshgrid(-M/2:M/2-1,-N/2:N/2-1);
        end


        function [holo_filtered,fx_max,fy_max,cir_mask] = spatial_filter(holo,M,N,visual,factor)
            % Compute Fourier Transfor hologram
            ft_holo = fftshift(fft2(fftshift(holo)));
        
            % filter DC term
            mask1 = ones(N,M);
            mask1(N/2-20:N/2+20,M/2-20:M/2+20)=0;
            ft_holo_I = ft_holo .* mask1;
        
            % filter reflection
            mask1 = ones(N,M);
            mask1(1,1)=0;
            ft_holo_I = ft_holo_I .* mask1;
            
            % select region of interest (ROI) to search the diffraction term.
            region_interest = ft_holo_I(1:N, 1:M/2);
            [maxValue, linearIndex] = max(abs(region_interest), [], 'all', 'linear');
            [fy_max, fx_max] = ind2sub(size(region_interest), linearIndex);
            
            % ciruclar mask to filter the +1 difraction term
             distance = sqrt((fx_max - N/2)^2+(fy_max - M/2)^2);
             resc = distance/factor;
            %resc = 80
            cir_mask = ones(N,M);
            for r=1:N
                for p=1:M
                    if sqrt((r-fy_max)^2+(p-fx_max)^2)>resc
                        cir_mask(r,p)=0;
                    end
                end
            end
    
            % Applying spatical filter and computing the filtered hologram
            ft_holo_filtered = ft_holo .* cir_mask;
            holo_filtered = fftshift(ifft2(fftshift(ft_holo_filtered)));
            
            if visual == 'Yes'
                figure,imagesc(log(abs(ft_holo).^2)),colormap(gray),title('FT Hologram'),daspect([1 1 1]) 
                %figure,imagesc(log(abs(region_interest).^2)),colormap(gray),title('ROI FT Hologram'),daspect([1 1 1]) 
                %figure,imagesc(cir_mask),colormap(gray),title('Circular Filter'),daspect([1 1 1])     
                figure,imagesc(log(abs(ft_holo_filtered).^2)),colormap(gray),title('FT Filtered Hologram'),daspect([1 1 1]) 
            end
        end


        function [ref_wave] = reference_wave(M,N,m,n,lambda,dxy,fx_max,fy_max,k,fx_0,fy_0)
            theta_x = asin((fx_0 - fx_max) * lambda / (M * dxy));
            theta_y = asin((fy_0 - fy_max) * lambda / (N * dxy));
            ref_wave = exp(1i * k * (sin(theta_x) * n * dxy + sin(theta_y) * m * dxy));
        end


        function [holo_filter,holo_FT,fx_max,fy_max] = spatialFilter_SIDHM(holo,M,N,visual,factor)
            fft_holo = fftshift(fft2(fftshift(holo(:,:,1)))); 
            figure,imagesc(log(abs(fft_holo).^2)),colormap(gray),title('FT Hologram'),daspect([1 1 1]) 
            
            [pointsY,pointsX] = ginput(2);
            x3 = round(pointsX(1));
            x4 = round(pointsX(2));
            y3 = round(pointsY(1));
            y4 = round(pointsY(2));
           
            mask = zeros(M,N);
            mask(x3:x4,y3:y4) = 1;
            fft_holo_I = fft_holo .* mask;
            figure,imagesc((abs( fft_holo_I).^0.1)),colormap(gray),title('FT Hologram'),daspect([1 1 1]) 
            
            % max values first peak
            maxValue_1 = max(max(abs(fft_holo_I)));
            [fy_max_1 fx_max_1] = find(abs(fft_holo_I) == maxValue_1)
            mask(fy_max_1 - 20:fy_max_1 + 20,fx_max_1 - 20:fx_max_1 + 20)=0;
            fx_max_L = fx_max_1;
            fy_max_L = fy_max_1;
            fft_holo_I = fft_holo_I .* mask;
            figure,imagesc(log(abs(fft_holo_I).^2)),colormap(gray),title('FT FIrst peak'),daspect([1 1 1])
            
            maxValue_1 = max(max(abs(fft_holo_I)));
            [fy_max_1 fx_max_1] = find(abs(fft_holo_I) == maxValue_1);
            fx_max_D = fx_max_1(1);
            fy_max_D = fy_max_1(1);
        
            fx_max = [fx_max_L,fx_max_D];
            fy_max = [fy_max_L,fy_max_D];
        
            %find the centers between both peaks 
            middlePoint_X = (fx_max_D + fx_max_L) / 2;
            middlePoint_Y = (fy_max_D + fy_max_L) / 2;
            
                    
            mask = zeros(M,N);
            mask(x3:x4,y3:y4) = 1;
            
            fft_filter_holo = fft_holo .* mask;
            holo_filter_1 = fftshift(ifft2(fftshift(fft_filter_holo)));
        
            fft_holo_2 = fftshift(fft2(fftshift(holo(:,:,2)))); 
            %fft_filter_holo_2 = fft_holo_2 .* filter;
            fft_filter_holo_2 = fft_holo_2 .* mask;
            holo_filter_2 = fftshift(ifft2(fftshift(fft_filter_holo_2)));
        
        
            holo_filter(:,:,1) = holo_filter_1;
            holo_filter(:,:,2) = holo_filter_2;
        
            holo_FT(:,:,1) = fft_filter_holo;
            holo_FT(:,:,2) = fft_filter_holo_2;
        
            if visual == 'Yes'
                figure,imagesc(log(abs(fft_filter_holo).^2)),colormap(gray),title('FT Filter Hologram'),daspect([1 1 1]) 
                figure,imagesc((abs(holo_filter_1).^2)),colormap(gray),title('Filter Hologram'),daspect([1 1 1]) 
            end
        end


        function [out] = ang_spectrum(field,z,lambda,dxy)
            %ANG_SPECTRUM Function to diffract a complex field using Angular Spectrum
            %method
            %   out = ANG_SPECTRUM(field,z,lambda,dx,dy)
            %
            %       field       complex field
            %       z           propagation distance
            %       lambda      wavelength
            %       dx/dy       sampling pitches
            
            [N,M] = size(field);
            [m,n] = meshgrid(1-M/2:M/2,1-N/2:N/2);
            
            dfx = 1 / (dxy * M);
            dfy = 1 / (dxy * N);

            % field = padarray(field,[floor(N/2) floor(M/2)]);
            field_spec = fftshift(fft2(fftshift(field)));
            % field_spec = padarray(field_spec,[floor(N/2) floor(M/2)]);
            
            phase = exp(1i * z * 2 * pi * sqrt((1 / lambda)^2 - ((m*dfx).^2 + (n*dfy).^2)));
            %phase = np.exp2(1j * z * pi * np.sqrt(np.power(1 / wavelength, 2) - (np.power(X * dfx, 2) + np.power(Y * dfy, 2))))
            
            % phase = padarray(phase,[floor(N/2) floor(M/2)]);
            
            out = ifftshift(ifft2(ifftshift(field_spec.*phase)));
            % out = out(floor(N/2)+1:floor(N/2)+N,floor(M/2)+1:floor(M/2)+M);
            
        end


        function [cf] = cost_function(seed_maxPeak,lambda,dxy,M,N,m,n,k,fx_0,fy_0,holo_filtered)
            cf = 0;
            fx_max = seed_maxPeak(1,1);
            fy_max = seed_maxPeak(1,2);
            k = 2*pi/lambda;
            theta_x = asin((fx_0 - fx_max) * lambda / (M * dxy));
            theta_y = asin((fy_0 - fy_max) * lambda / (N * dxy));
            ref_wave = exp(1i * k * (sin(theta_x) * n * dxy + sin(theta_y) * m * dxy));
        
            holo_rec = holo_filtered .* ref_wave;
            phase = angle(holo_rec);
        
            phase_save = uint8(255 * mat2gray(phase));
            ib = imbinarize(phase_save, 0.1);
            cf = M*N - sum(ib(:));

            %target = ones(M, N); % Supongamos que el objetivo es una imagen completamente binarizada (blanca)
            %cf_particleswarm = double(target(:)) - double(ib(:));
        end


        function [diffeAverage] = profiles(phase,num_zones, visual)
            lateralDistance = cell(num_zones, 1);
            phaseValues = cell(num_zones, 1);

            for cont = 1:num_zones
                fprintf('Select two points for zone %d:\n', cont);
    
                % Allow user to select two points on the image
                [x, y] = ginput(2);

                % Draw a line between the selected points
                hold on;
                plot(x, y, 'r', 'LineWidth', 2);
                hold off;

                % Calculate the phase values along the line based on 2D phase values
                numPoints = round(max(abs(diff(x)), abs(diff(y)))) + 1;
                xValues = linspace(x(1), x(2), numPoints);
                yValues = linspace(y(1), y(2), numPoints); 

                phasVal = interp2(double(phase), xValues, yValues, 'linear');
                phasVal = phasVal(:);

                lateralDistance{cont} = (1:numPoints)'; 
                phaseValues{cont} = phasVal; 
            end

            if visual == 'true'
                % Plotting pixel values for all regions
                figure; % Open a new figure window
                hold on;
                for cont = 1:num_zones
                    plot(lateralDistance{cont}, phaseValues{cont}, '-', 'DisplayName', sprintf('Zone %d', cont));
                end
                xlabel('Lateral Position [um]');
                ylabel('Phase Value [rad]');
                legend show; 
                grid on; 
                hold off;
            end
            
            table = lateralDistance;

            for cont = 1:num_zones
                pixelValues = phaseValues{cont};
                sortedValues = sort(pixelValues);                
                percentage = 0.1;
            
                numElements = round(percentage * length(sortedValues));
                highMean = mean(sortedValues(end-numElements+1:end)); 
                lowMean = mean(sortedValues(1:numElements)); 
                    
                differences(cont) = abs(abs(highMean) - abs(lowMean));
                fprintf('Phase diference zone %d: %.2f\n', cont, differences(cont));
                
            end

            diffeAverage = mean(differences(:));
            fprintf('Average phase difference %.2f [rad] \n' , diffeAverage);
            % plot phase diference 
            % figure; 
            % bar(differences); 
            % xlabel('zone');
            % ylabel('Phase diference');
            % grid on;

        end


        function [diffeAverage] = circulos(phase,circulos)
            numCirculos = circulos;
            disp('Dibuja un círculo sobre la imagen y luego presiona Enter para continuar.');
            for i = 1:numCirculos
                cir = drawcircle;



                waitforbuttonpress;
                x_origen = cir.Center(1);
                y_origen = cir.Center(2);
                radio = cir.Radius;

                [fila, col, ~] = size(phase);
                [X, Y] = meshgrid(1:col, 1:fila);
                distancia = sqrt((X - x_origen).^2 + (Y - y_origen).^2);

                mas_circulo = distancia <= radio;

                radioAnillo1 = radio + 15;
                radioAnillo2 = radioAnillo1 + 20;

                masAnillo1 = distancia <= radioAnillo1 & distancia > radio;
                masAnillo2 = distancia <= radioAnillo2 & distancia > radioAnillo1;

                valoresCirculo = phase(mas_circulo);
                valoresAnillo1 = phase(masAnillo1);
                valoresAnillo2 = phase(masAnillo2);

                valoresAnillo = [valoresAnillo1; valoresAnillo2];

                promedioCirculo = mean(valoresCirculo);  % Promedio dentro del círculo rojo
                promedioAnillo = mean(valoresAnillo);  % Promedio en el área entre los dos círculos azules
                diferencia(i) = abs(abs(promedioCirculo) - abs( promedioAnillo));

                fprintf('Círculo %d:\n', i);
                fprintf('  Diferencia: %.2f ', i, diferencia(i));
                % fprintf('  Promedio de fase góbulo: %.2f\n', i, promedioCirculo);
                % fprintf('  Promedio de fase fondo: %.2f\n', i, promedioAnillo);

                rectangle('Position', [x_origen - radio, y_origen - radio, 2*radio, 2*radio], ...
                    'Curvature', [1, 1], 'EdgeColor', 'r', 'LineWidth', 2);
                rectangle('Position', [x_origen - radioAnillo1, y_origen - radioAnillo1, 2*radioAnillo1, 2*radioAnillo1], ...
                    'Curvature', [1, 1], 'EdgeColor', 'b', 'LineWidth', 2);
                rectangle('Position', [x_origen - radioAnillo2, y_origen - radioAnillo2, 2*radioAnillo2, 2*radioAnillo2], ...
                    'Curvature', [1, 1], 'EdgeColor', 'b', 'LineWidth', 2);
            end

            hold off;
            diffeAverage = mean(diferencia(:));
            fprintf('Average phase difference %.2f [rad] \n' , diffeAverage);
        end


        function [diffeAverage,numero_globulos] = segmentacion(phase)
            umbral = 0.2;
            imagenBinaria = umbral > phase;
            % imshow(imagenBinaria);
            stats = regionprops(imagenBinaria, 'Centroid', 'MajorAxisLength', 'BoundingBox', 'Area');
            [altura, ancho] = size(phase);

            margen = 50;

            valores_cruz_por_centroid = {};

            contador = 0;

            acumulado_fase = 0;
            numero_globulos = 0;

            porcentaje = 0.15;

            figure(1);
            hold on;

            centroides = cat(1, stats.Centroid);
            plot(centroides(:,1), centroides(:,2), 'bo', 'MarkerSize', 2, 'MarkerFaceColor', 'yellow');

            disp('Haga clic sobre los glóbulos que desea procesar. Presione ENTER cuando haya terminado.');


            [x_usuario, y_usuario] = ginput;

            distancias = sqrt((centroides(:,1) - x_usuario').^2 + (centroides(:,2) - y_usuario').^2);
            [~, indices_seleccionados] = min(distancias, [], 1);

            for k = indices_seleccionados
                centroid = stats(k).Centroid;
                x = centroid(1);
                y = centroid(2);

                contador = contador + 1;

                diametro = stats(k).MajorAxisLength;

                longitud = diametro + 30; %%%Majo este es para la longitud de la linea

                plot([x - longitud/2, x + longitud/2], [y, y], 'r-', 'LineWidth', 2);
                plot([x, x], [y - longitud/2, y + longitud/2], 'r-', 'LineWidth', 2);

                plot(x, y, 'b.', 'MarkerSize', 10);

                text(x, y - 10, sprintf('%d', contador), 'Color', 'yellow', 'FontSize', 12, 'HorizontalAlignment', 'center');

                x_horizontal = round(x - longitud/2):round(x + longitud/2);
                y_horizontal = repmat(round(y), size(x_horizontal));

                valores_horizontal = phase(sub2ind(size(phase), y_horizontal, x_horizontal));

                y_vertical = round(y - longitud/2):round(y + longitud/2);
                x_vertical = repmat(round(x), size(y_vertical));

                valores_vertical = phase(sub2ind(size(phase), y_vertical, x_vertical));

                valores_horizontal = sort(valores_horizontal);
                valores_vertical = sort(valores_vertical);

                num_elementos_horizontal = round(porcentaje * length(valores_horizontal));
                num_elementos_vertical = round(porcentaje * length(valores_vertical));

                promedio_bajo_horizontal = mean(valores_horizontal(1:num_elementos_horizontal));  % Promedio de los primeros 'num_elementos' valores
                promedio_alto_horizontal = mean(valores_horizontal(end-num_elementos_horizontal+1:end));  % Promedio de los últimos 'num_elementos' valores

                promedio_bajo_vertical = mean(valores_vertical(1:num_elementos_vertical));  % Promedio de los primeros 'num_elementos' valores
                promedio_alto_vertical = mean(valores_vertical(end-num_elementos_vertical+1:end));  % Promedio de los últimos 'num_elementos' valores

                diferencia_horizontal = abs(abs(promedio_alto_horizontal) - abs(promedio_bajo_vertical));

                diferencia_vertical = abs(abs(promedio_alto_vertical) - abs(promedio_bajo_horizontal));

                diffeAverage = mean([diferencia_horizontal, diferencia_vertical]);  % Promedio de las diferencias

                acumulado_fase = acumulado_fase + diffeAverage;

                numero_globulos = numero_globulos + 1;
                valores_cruz_por_centroid{contador} = struct(...
                    'horizontal', valores_horizontal, ...
                    'vertical', valores_vertical, ...
                    'promedio_bajo_horizontal', promedio_bajo_horizontal, ...
                    'promedio_alto_horizontal', promedio_alto_horizontal, ...
                    'promedio_bajo_vertical', promedio_bajo_vertical, ...
                    'promedio_alto_vertical', promedio_alto_vertical, ...
                    'diferencia_horizontal', diferencia_horizontal, ...
                    'diferencia_vertical', diferencia_vertical, ...
                    'promedio_fase', diffeAverage ...
                    );
            end
            hold off;
            disp(['Número total glóbulos: ', num2str(numero_globulos)]);
            disp(['Promedio global de fase: ', num2str(diffeAverage)]);

        end

        function deltaPhase = calculateProfileDifference(profileValues, percentage)

            % Convert profile to a column vector
            profileValues = profileValues(:);
        
            % Remove invalid values
            profileValues = profileValues(~isnan(profileValues));
        
            if isempty(profileValues)
                deltaPhase = NaN;
                return;
            end
        
            % Sort phase values
            sortedValues = sort(profileValues);
        
            % Determine the number of values used at each extreme
            numElements = max( ...
                1, ...
                round(percentage * numel(sortedValues)));
        
            % Calculate the mean low and high phase levels
            lowMean = mean(sortedValues(1:numElements));
        
            highMean = mean( ...
                sortedValues(end - numElements + 1:end));
        
            % Calculate phase difference
            deltaPhase = abs(highMean - lowMean);
        end

        function [resultsTable, globalAverage] = ...
        diagonalRBCProfiles(phase, numRBCs, visual)

            % Validate input data
            if ~ismatrix(phase) || ~isnumeric(phase)
                error('The phase input must be a two-dimensional numeric matrix.');
            end
        
            if numRBCs < 1 || mod(numRBCs, 1) ~= 0
                error('numRBCs must be a positive integer.');
            end
        
            % Percentage used to estimate low and high phase levels
            percentage = 0.10;
        
            % Initialize result vectors
            diagonal1Difference = zeros(numRBCs, 1);
            diagonal2Difference = zeros(numRBCs, 1);
            meanPhaseDifference = zeros(numRBCs, 1);
        
            centerX = zeros(numRBCs, 1);
            centerY = zeros(numRBCs, 1);
            radiusValues = zeros(numRBCs, 1);
        
            % Store profiles for optional visualization
            profileDiagonal1 = cell(numRBCs, 1);
            profileDiagonal2 = cell(numRBCs, 1);
        
            % Display phase image
            figure('Name', 'RBC selection');
            imagesc(phase);
            axis image;
            colormap(gray);
            colorbar;
            hold on;
        
            title({ ...
                'Draw one circle around each RBC', ...
                'Double-click inside the circle to confirm the selection'});
        
            [numberOfRows, numberOfColumns] = size(phase);
        
            for index = 1:numRBCs
        
                fprintf( ...
                    'Draw a circle around RBC %d and double-click to confirm.\n', ...
                    index);
        
                % Manual circular selection
                circleROI = drawcircle( ...
                    'Color', 'yellow', ...
                    'LineWidth', 1.5);
        
                wait(circleROI);
        
                center = circleROI.Center;
                radius = circleROI.Radius;
        
                xCenter = center(1);
                yCenter = center(2);
        
                centerX(index) = xCenter;
                centerY(index) = yCenter;
                radiusValues(index) = radius;
        
                % Use the circle diameter as profile length
                halfLength = radius;
        
                % First diagonal: upper-left to lower-right
                x1Start = xCenter - halfLength / sqrt(2);
                y1Start = yCenter - halfLength / sqrt(2);
        
                x1End = xCenter + halfLength / sqrt(2);
                y1End = yCenter + halfLength / sqrt(2);
        
                % Second diagonal: lower-left to upper-right
                x2Start = xCenter - halfLength / sqrt(2);
                y2Start = yCenter + halfLength / sqrt(2);
        
                x2End = xCenter + halfLength / sqrt(2);
                y2End = yCenter - halfLength / sqrt(2);
        
                % Keep line coordinates inside the image
                x1Start = max(1, min(numberOfColumns, x1Start));
                x1End   = max(1, min(numberOfColumns, x1End));
        
                y1Start = max(1, min(numberOfRows, y1Start));
                y1End   = max(1, min(numberOfRows, y1End));
        
                x2Start = max(1, min(numberOfColumns, x2Start));
                x2End   = max(1, min(numberOfColumns, x2End));
        
                y2Start = max(1, min(numberOfRows, y2Start));
                y2End   = max(1, min(numberOfRows, y2End));
        
                % Number of interpolation points
                numPoints1 = max( ...
                    2, ...
                    round(max( ...
                        abs(x1End - x1Start), ...
                        abs(y1End - y1Start))) + 1);
        
                numPoints2 = max( ...
                    2, ...
                    round(max( ...
                        abs(x2End - x2Start), ...
                        abs(y2End - y2Start))) + 1);
        
                % Generate line coordinates
                xDiagonal1 = linspace(x1Start, x1End, numPoints1);
                yDiagonal1 = linspace(y1Start, y1End, numPoints1);
        
                xDiagonal2 = linspace(x2Start, x2End, numPoints2);
                yDiagonal2 = linspace(y2Start, y2End, numPoints2);
        
                % Extract phase values using interpolation
                valuesDiagonal1 = interp2( ...
                    double(phase), ...
                    xDiagonal1, ...
                    yDiagonal1, ...
                    'linear');
        
                valuesDiagonal2 = interp2( ...
                    double(phase), ...
                    xDiagonal2, ...
                    yDiagonal2, ...
                    'linear');
        
                % Remove possible NaN values
                valuesDiagonal1 = valuesDiagonal1(~isnan(valuesDiagonal1));
                valuesDiagonal2 = valuesDiagonal2(~isnan(valuesDiagonal2));
        
                if isempty(valuesDiagonal1) || isempty(valuesDiagonal2)
                    warning( ...
                        'RBC %d contains an invalid diagonal profile.', ...
                        index);
        
                    diagonal1Difference(index) = NaN;
                    diagonal2Difference(index) = NaN;
                    meanPhaseDifference(index) = NaN;
        
                    continue;
                end
        
                profileDiagonal1{index} = valuesDiagonal1;
                profileDiagonal2{index} = valuesDiagonal2;
        
                % Calculate phase difference for first diagonal
                diagonal1Difference(index) = ...
                    functions_anemia.calculateProfileDifference( ...
                        valuesDiagonal1, percentage);
        
                % Calculate phase difference for second diagonal
                diagonal2Difference(index) = ...
                    functions_anemia.calculateProfileDifference( ...
                        valuesDiagonal2, percentage);
        
                % Average both diagonal measurements
                meanPhaseDifference(index) = mean( ...
                    [diagonal1Difference(index), ...
                     diagonal2Difference(index)], ...
                    'omitnan');
        
                % Draw diagonal lines
                plot( ...
                    [x1Start, x1End], ...
                    [y1Start, y1End], ...
                    'r-', ...
                    'LineWidth', 1.5);
        
                plot( ...
                    [x2Start, x2End], ...
                    [y2Start, y2End], ...
                    'c-', ...
                    'LineWidth', 1.5);
        
                % Draw center
                plot( ...
                    xCenter, ...
                    yCenter, ...
                    'y.', ...
                    'MarkerSize', 14);
        
                % Add RBC number
                text( ...
                    xCenter, ...
                    yCenter - radius - 5, ...
                    sprintf('%d', index), ...
                    'Color', 'yellow', ...
                    'FontSize', 11, ...
                    'FontWeight', 'bold', ...
                    'HorizontalAlignment', 'center');
        
                fprintf( ...
                    ['RBC %d | Diagonal 1 = %.4f rad | ' ...
                     'Diagonal 2 = %.4f rad | Mean = %.4f rad\n'], ...
                    index, ...
                    diagonal1Difference(index), ...
                    diagonal2Difference(index), ...
                    meanPhaseDifference(index));
            end
        
            hold off;
        
            % Create result table
            RBC = (1:numRBCs)';
        
            resultsTable = table( ...
                RBC, ...
                centerX, ...
                centerY, ...
                radiusValues, ...
                diagonal1Difference, ...
                diagonal2Difference, ...
                meanPhaseDifference, ...
                'VariableNames', { ...
                    'RBC', ...
                    'CenterX_px', ...
                    'CenterY_px', ...
                    'Radius_px', ...
                    'DeltaPhaseDiagonal1_rad', ...
                    'DeltaPhaseDiagonal2_rad', ...
                    'MeanDeltaPhase_rad'});
        
            % Calculate global average
            globalAverage = mean(meanPhaseDifference, 'omitnan');
        
            % Optional profile visualization
            if visual
        
                figure('Name', 'Diagonal phase profiles');
                tiledlayout('flow');
        
                for index = 1:numRBCs
        
                    if isempty(profileDiagonal1{index}) || ...
                            isempty(profileDiagonal2{index})
                        continue;
                    end
        
                    nexttile;
                    hold on;
        
                    plot( ...
                        profileDiagonal1{index}, ...
                        'LineWidth', 1.2, ...
                        'DisplayName', 'Diagonal 1');
        
                    plot( ...
                        profileDiagonal2{index}, ...
                        'LineWidth', 1.2, ...
                        'DisplayName', 'Diagonal 2');
        
                    xlabel('Position [pixel]');
                    ylabel('Phase [rad]');
                    title(sprintf('RBC %d', index));
                    legend('Location', 'best');
                    grid on;
                    hold off;
                end
            end
        
            fprintf( ...
                '\nGlobal average phase difference: %.4f rad\n', ...
                globalAverage);
        end

    end
end


