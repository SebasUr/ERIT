function output = vortexLegendre(hologram, wavelenth, dxy, medFilt, NoPistonCompensation, UsePCA, limit)
    % Verified that the hologram has a square size
    hologram = hologram(:,:,1);
    if size(hologram,2) ~= size(hologram,1)
        lim=round(size(hologram,2)-size(hologram,1))/2;
        hologram = hologram(:,lim:end-lim-1);
    end
    
    [N, M]=size(hologram);
    [n,m] = meshgrid(-M/2:M/2-1,-N/2:N/2-1);

    % Parameters 
    k = 2 * pi / wavelenth;
    fx_0 = M/2;
    fy_0 = N/2;

    % Filter oreder +1
    [holo_filtered, fxOverMax, fyOverMax, cir_mask] = ...
        functions_vortexLegendre.spatial_filter(hologram,M,N,'Not',3);

    % Vortex subpixel frequencies calculation
    ft_holo = log(abs(fftshift(fft2(fftshift(hologram)))).^2);
    field = medfilt2(ft_holo, [medFilt, medFilt], 'symmetric');
    [positions] = functions_vortexLegendre.vortexCompensation(field, fxOverMax, fyOverMax);
    

    % Tilt compensation aberration using optical vortex
    [ref_wave] = functions_vortexLegendre.reference_wave...
    (M,N,m,n,wavelenth,dxy,positions(1),positions(2),k,fx_0,fy_0);
    field_compensate = ref_wave.*holo_filtered;

    % PCA + Legendre
    [phaseCorrected, Legendre_Coefficients] = ...
        functions_vortexLegendre.LegendreCompensation(field_compensate, limit, NoPistonCompensation, UsePCA);
    
    %figure,imagesc(angle(phaseCorrected)),colormap(gray),colorbar
    %title('Phase compensated (Vortex + Legendre)'),daspect([1 1 1])

    % Vortex + piston
    gridSize = size(angle(field_compensate),1);
    [X, Y] =  meshgrid(-1:(2 / gridSize):(1 - 2 / gridSize), ...
        -1:(2 / gridSize):(1 - 2 / gridSize));
    dA=(2 / gridSize) ^ 2;
    
    order = 2:6;
    [polynomials] = functions_vortexLegendre.squareLegendrefitting(order, X, Y);
    Legendres = reshape(polynomials, [size(polynomials, 1)*size(polynomials, 2) ...
        size(polynomials, 3)]);
    zProds = Legendres.'* Legendres * dA;
    zNorm = bsxfun(@rdivide, Legendres, sqrt(diag(zProds).'));
    Legendres = (bsxfun(@times, (ones(size(order))').', zNorm));
    Legendres_norm_const =sum(Legendres.^2,1)*dA;
    
    WavefrontReconstructed_Vect = sum(repmat(Legendre_Coefficients(2:size(order,2)+1)./ ...
        sqrt(Legendres_norm_const(1:size(order,2))),[size(Legendres,1) 1]).*Legendres(:,1:size(order,2)),2);
    
    WavefrontReconstructed = reshape(WavefrontReconstructed_Vect, ...
        [size(polynomials, 1) size(polynomials, 2)]);
    
    compensatedHologram = abs(field_compensate) .* (exp(1i.* angle(field_compensate))./ ...
        exp((1i).*WavefrontReconstructed));

    output = compensatedHologram;


end