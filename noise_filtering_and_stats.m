function filter_signal(inputFile, outputFile, filterType, cutoffFreq)
    % Reading the data
    data = readtable(inputFile);
    signal = data.("Sensor Value");
    time = data.("Time");
    fs = 800;

    % Filter design
    switch lower(filterType)
        case 'butterworth'
            [b, a] = butter(4, cutoffFreq/(fs/2), 'low');
            filtered = filtfilt(b, a, signal);
        case 'chebyshev1'
            [b, a] = cheby1(4, 0.5, cutoffFreq/(fs/2), 'low');
            filtered = filtfilt(b, a, signal);
        case 'moving average'
            windowSize = round(fs / cutoffFreq);
            filtered = movmean(signal, windowSize);
        otherwise
            error('Unsupported filter type');
    end

    % Filtered value
    data.("Filtered Sensor Value") = filtered;
    writetable(data, outputFile);
    
    % Analysis
    fid = fopen("filter_analysis.txt", "w");
    fprintf(fid, "Filter type: %s\n", filterType);
    fprintf(fid, "Cutoff frequency: %.2f Hz\n", cutoffFreq);
    fprintf(fid, "Original mean: %.4f\n", mean(signal));
    fprintf(fid, "Filtered mean: %.4f\n", mean(filtered));
    fprintf(fid, "Original std: %.4f\n", std(signal));
    fprintf(fid, "Filtered std: %.4f\n", std(filtered));
    fprintf(fid, "Noise energy removed: %.4f\n", sum((signal - filtered).^2));
    fclose(fid);

    % Плот
    figure;
    plot(time, signal, 'r-', 'DisplayName', 'Original');
    hold on;
    plot(time, filtered, 'b-', 'DisplayName', 'Filtered');
    legend;
    title('Signal Filtering');
    xlabel('Time (s)');
    ylabel('Value');
    grid on;
    saveas(gcf, 'filtered_plot.png');
end
