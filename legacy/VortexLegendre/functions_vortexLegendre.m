classdef functions_vortexLegendre
    methods(Static)

         % Function to load a hologram 
        function [holo,M,N,m,n] = holo_read(filename)
            holo = double(imread(filename));
            holo = holo(:,:,1);
            [N,M] = size(holo);
            [n,m] = meshgrid(-M/2:M/2-1,-N/2:N/2-1);
        end
        
        % Spatial filert +1 order        
        function [holo_filtered,fx_max,fy_max,cir_mask] = spatial_filter(holo,M,N,visual,factor)
            % Compute Fourier Transfor hologram
            ft_holo = fftshift(fft2(fftshift(holo)));
            ft_holo(1:5,1:5)=0;
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
                % phase_save =  mat2gray(log(angle(ft_holo).^2));
                % folder_p = fullfile(folder_phase,strcat(nameNoExt, "_VortexLegPis_", '.bmp'));
                %saveas(gcf, 'espectro.bmp');
                % imwrite(phase_save, 'espectro.bmp');
                % figure,imagesc(log(abs(region_interest).^2)),colormap(gray),title('ROI FT Hologram'),daspect([1 1 1]) 
                figure,imagesc(cir_mask),colormap(gray),title('Circular Filter'),daspect([1 1 1])
                
                figure,imagesc(log(abs(ft_holo_filtered).^2)),colormap(gray),title('FT Filtered Hologram'),daspect([1 1 1]) 
                %saveas(gcf, 'mask.bmp');
            end
        end
        
        % Vortex compensation
        function [positions] = vortexCompensation ...
            (field, fxOverMax, fyOverMax)

            cropVortex = 5; % Pixels for interpolation.
            factorOverInterpolation = 55;
            sd = field(fyOverMax-cropVortex:fyOverMax+cropVortex-1,...
                fxOverMax-cropVortex:fxOverMax+cropVortex-1);
            
            
            sd_crop = functions_vortexLegendre.hilbertTransform2D(sd,1);
            
            sz = size(abs(sd_crop));
            xg = 1:sz(1);
            yg = 1:sz(2);
            
            F = griddedInterpolant({xg,yg}, ...
                real(sd_crop));
            F2 = griddedInterpolant({xg,yg}, ...
                imag(sd_crop));
            
            xq = (0:1/factorOverInterpolation:(sz(1)-1/(factorOverInterpolation)))';
            yq = (0:1/factorOverInterpolation:(sz(2)-1/(factorOverInterpolation)))';
            
            vq = double(F({xq,yq}));
            vq2 = double(F2({xq,yq}));
            
            
            psi = angle(vq+(1i.*vq2));
            [n1,m1]=size(psi);
            Ml=zeros(n1,m1);
            
            % Application of the residue theorem
            M1=Ml; M2=Ml; M3=Ml; M4=Ml; M5=Ml; M6=Ml; M7=Ml; M8=Ml;
            
            Y1=1:n1-2; 
            Y2=2:n1-1;
            Y3=3:n1;
            X1=1:m1-2; 
            X2=2:m1-1; 
            X3=3:m1;
            
            M1(Y2,X2)=psi(Y1,X1); M2(Y2,X2)=psi(Y1,X2);
            M3(Y2,X2)=psi(Y1,X3); M4(Y2,X2)=psi(Y2,X3);
            M5(Y2,X2)=psi(Y3,X3); M6(Y2,X2)=psi(Y3,X2);
            M7(Y2,X2)=psi(Y3,X1); M8(Y2,X2)=psi(Y2,X1);
            
            D1=wrapToPi(M2-M1); D2=wrapToPi(M3-M2);
            D3=wrapToPi(M4-M3); D4=wrapToPi(M5-M4);
            D5=wrapToPi(M6-M5); D6=wrapToPi(M7-M6);
            D7=wrapToPi(M8-M7); D8=wrapToPi(M1-M8);
            
            Ml=D1+D2+D3+D4+D5+D6+D7+D8;
            Ml=fftshift(Ml/(2*pi));
            Ml(70:end, 70:end) = 0;
            Ml= ifftshift(Ml);
            
            [~, linearIndex] = min(Ml, [], 'all', 'linear');
            [yOverInterpolVortex, xOverInterpolVortex] = ind2sub(size(Ml), linearIndex);
            
            positions = [];
            positions=[positions; (xOverInterpolVortex/factorOverInterpolation)+((fxOverMax)-cropVortex-2),...
                (yOverInterpolVortex/factorOverInterpolation)+((fyOverMax)-cropVortex-2)];
            
        end

        % Digital reference wave
        function [ref_wave] = reference_wave(M,N,m,n,lambda,dxy,fx_max,fy_max,k,fx_0,fy_0)
            theta_x = asin((fx_0 - fx_max) * lambda / (M * dxy));
            theta_y = asin((fy_0 - fy_max) * lambda / (N * dxy));
            ref_wave = exp(1i * k * (sin(theta_x) * n * dxy + sin(theta_y) * m * dxy));
        end

        % squareLegendrefitting
        function polynomials = squareLegendrefitting(order, x, y)
            all_polynomials = {1;
                x;
                y;
                ((3.*(x.^2))-1)/2;
                x.*y;
                ((3.*(y.^2))-1)/2;
                (x.*((5.*(x.^2))-3))/2;
                (y.*((3.*(x.^2))-1))/2;
                (x.*((3.*(y.^2))-1))/2;
                (y.*((5.*(y.^2))-3))/2;
                ((35.*(x.^4))-(30.*(x.^2))+3)/8;
                (x.*y.*((5.*(x.^2))-3))/2;
                (((3.*(y.^2))-1).*((3.*(x.^2))-1))/4;
                (x.*y.*((5.*(y.^2))-3))/2;
                ((35.*(y.^4))-(30.*(y.^2))+3)/8;};
            
            polynomials = zeros(size(x));
            for i = 1:length(order)
                polynomials(:,:,i) = all_polynomials{order(i)};
            end
        end

        % Hilber transform 
        function cuadrature = hilbertTransform2D(c,HilbertOrEnergyOperator)
            [NR, NC]=size(c);
            [u,v]=meshgrid(1:NC, 1:NR);
            u0=floor(NC/2)+1;
            v0=floor(NR/2)+1;
            
            u=u-u0;
            v=v-v0;
            
            H=(u+1i*v)./abs(u+1i*v);
            H(v0, u0)=0; 
            
            C=fft2(c);
            
            if HilbertOrEnergyOperator
                CH=C.*ifftshift(H);
            else
                CH=C.*ifftshift(1i.*H);
            end
            
            cuadrature = conj(ifft2(CH));
        end

        % Legendre Compensation
        function [compensatedHologram, Legendre_Coefficients] ...
            = LegendreCompensation(field_compensate, limit, NoPistonCompensation, UsePCA)
        
            fftFieldCompensated = fftshift(fft2(ifftshift(field_compensate)));
        
            % Reduced region calculation
            [A, B] = size(fftFieldCompensated);
            fftFieldCompensated = ...
                fftFieldCompensated(round(A/2)+1-limit:round(A/2)+limit, ...
                                    round(B/2)+1-limit:round(B/2)+limit);
            square = ifftshift(ifft2(fftshift(fftFieldCompensated)));
        
            % PCA / SVD dominant component
            if UsePCA
                disp('Estoy dentro UPCA')
                [U, S, V] = svd((square));
                numComponentes = 1;
                dominantComponent = U(:, 1:numComponentes) * ...
                    S(1:numComponentes, 1:numComponentes) * V(:, 1:numComponentes)';
                dominantComponent = unwrap_phase(angle(dominantComponent));
            else
                dominantComponent = unwrap_phase(angle(square));
            end
        
            % Legendre stuff
            gridSize = size(dominantComponent,1);
            [X, Y] =  meshgrid(-1:(2 / gridSize):(1 - 2 / gridSize), ...
                               -1:(2 / gridSize):(1 - 2 / gridSize));
            dA = (2 / gridSize)^2;
            order = 1:10;
            polynomials = functions_vortexLegendre.squareLegendrefitting(order, X, Y);
        
            Legendres = reshape(polynomials, [size(polynomials, 1)*size(polynomials, 2) ...
                                              size(polynomials, 3)]);
            zProds = Legendres.' * Legendres * dA;
            zNorm  = bsxfun(@rdivide, Legendres, sqrt(diag(zProds).'));
            Legendres = bsxfun(@times, ones(1, numel(order)), zNorm);
            Legendres_norm_const = sum(Legendres.^2, 1) * dA;
        
            phaseVector = reshape(dominantComponent, ...
                                  [size(polynomials, 1)*size(polynomials, 2) 1]);
            Legendre_Coefficients = sum(bsxfun(@times, Legendres, phaseVector), 1) * dA;
        
            % --- PISTON CORRECTION LOGIC ---
            if NoPistonCompensation
                WavefrontReconstructed_Vect = sum( ...
                    repmat(Legendre_Coefficients(2:end) ./ ...
                           sqrt(Legendres_norm_const(2:end)), ...
                           [size(Legendres,1) 1]) .* Legendres(:,2:end), 2);
        
                WavefrontReconstructed = reshape(WavefrontReconstructed_Vect, ...
                    [size(polynomials, 1) size(polynomials, 2)]);
        
                compensatedHologram = exp(1i .* angle(square)) ./ ...
                                      exp(1i .* WavefrontReconstructed);
        
            else
                evaluationPiston = 1;
                sizePistonEvaluation = -pi:(pi/6):pi;
                variance = zeros(numel(sizePistonEvaluation), 1);  
        
                for i = sizePistonEvaluation
                    Legendre_Coefficients(1) = i;
                    WavefrontReconstructed_Vect = sum( ...
                        repmat(Legendre_Coefficients(1:end) ./ ...
                               sqrt(Legendres_norm_const(1:end)), ...
                               [size(Legendres,1) 1]) .* Legendres(:,1:end), 2);
        
                    WavefrontReconstructed = reshape(WavefrontReconstructed_Vect, ...
                        [size(polynomials, 1) size(polynomials, 2)]);
        
                    compensatedHologram_tmp = exp(1i .* angle(square)) ./ ...
                                              exp(1i .* WavefrontReconstructed);
        
                    evaluationPistonPhase = angle(compensatedHologram_tmp);
                    variance(evaluationPiston) = var(evaluationPistonPhase(:));
                    evaluationPiston = evaluationPiston + 1;
                end
        
                % Determining the piston value for proper compensation using variance analysis
                [~, posVz] = min(variance);
                Legendre_Coefficients(1) = sizePistonEvaluation(posVz);
        
                WavefrontReconstructed_Vect = sum( ...
                    repmat(Legendre_Coefficients(1:end) ./ ...
                           sqrt(Legendres_norm_const(1:end)), ...
                           [size(Legendres,1) 1]) .* Legendres(:,1:end), 2);
        
                WavefrontReconstructed = reshape(WavefrontReconstructed_Vect, ...
                    [size(polynomials, 1) size(polynomials, 2)]);
        
                compensatedHologram = exp(1i .* angle(square)) ./ ...
                                      exp(1i .* WavefrontReconstructed);
            end
        end


    end
end
