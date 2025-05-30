import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, Tk
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import mplcursors
from scipy.stats import linregress

def read_data():
    filepath = filedialog.askopenfilename(title="Select a txt file", filetypes=[("txt files", "*.txt")])
    if not filepath:
        return None
    df = pd.read_csv(filepath, delimiter='\t', header=None, dtype='float32')
    return df

def convert_np(an_array):
    values = [float(x) for x in an_array.split()]
    an_array = np.array(values)
    return an_array

def analyze_peaks():
    df = read_data()
    if df is None:
        return



    def peakL(x, A1, P1, W1):
        return A1 * W1 / ((np.pi) * ((x - P1)**2 + W1**2))

    def fit_L(x, A, P, W): # type: ignore
        term_1 = peakL(x, A, P, W)
        term_2 = backline(x)
        return term_1 + term_2

    def backline(x):
        backliney = (y_last - y_first) / (x_last - x_first) * (x - x_first) + y_first - const
        return backliney

    def normalize_data(data): # WAS USED DUE TO TKINTER ERROR, not anymore 
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    
    def gaussian(x, amp, mu, sigma):
        return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

#gaussian: A single Gaussian curve where:

#amp is the amplitude (height of the peak)
#centre is the center position of the peak
#width is related to the full width at half maximum (FWHM = 2*sqrt(ln(2))*width)


    #double_gaussian: Sum of two Gaussian curves

    def double_gaussian(x, amp1, mu1, sigma1, amp2, mu2, sigma2):
        return gaussian(x, amp1, mu1, sigma1) + gaussian(x, amp2, mu2, sigma2)

    all_peaks = []
    peaks = dict()
    areas = dict()
    AD_AG_ratio = []
    AD_AG_ratio_err = []
    


    shifts = np.array(df.iloc[0, 2:].values)
    intensities = np.array(df.iloc[1:, 2:].values)
    y_coordinates = np.array(df.iloc[1:, 0].values)
    x_coordinates = np.array(df.iloc[1:, 1].values)

    fixed_ranges = {
        'D': (1250, 1450),  # Example values
        'G': (1450, 1700),
        '2D': (2600, 2900)
    }
    
    sample_id = simpledialog.askstring("Sample ID", "Please enter the sample ID:")

    while True:
        peak = simpledialog.askstring("Peaks", 'Name a peak you want to analyse. If you have inputted all the peaks, input "No":')
        if peak == "No" or peak is None:
            break
        all_peaks.append(peak)

    first_try = True
    while first_try:
        while True:
            peak_index = simpledialog.askfloat("Peak Index", "Type the index of the peak you want as an example. Note down your estimate of peak area and FWHM.")
            if peak_index is not None and 0 <= peak_index < len(intensities):
                plt.figure()
                plt.plot(shifts, intensities[int(peak_index)])
                plt.title(f'Sample ID: {sample_id}')
                plt.show()
                if first_try:
                    try_again = simpledialog.askstring("Choose Again", "Would you like to choose another index? (Yes/No)")
                    if try_again.lower() != 'yes': # type: ignore
                        first_try = False
                    break
            else:
                print("Invalid index entered.")
                first_try = False
                break

    D_area_guess = simpledialog.askfloat("D Peak Area Guess", "Input your estimate for the area under the D peak")
    D_width_guess = simpledialog.askfloat("D Peak Width Guess", "Input your estimate for the width of the D Peak")

    G_area_guess = simpledialog.askfloat("G Peak Area Guess", "Input your estimate for the area under the G peak")
    G_width_guess = simpledialog.askfloat("G Peak Width Guess", "Input your estimate for the width of the G Peak")

    TwoD_area_guess = simpledialog.askfloat("2D Peak Area Guess", "Input your estimate for the area under the 2D peak")
    TwoD_width_guess = simpledialog.askfloat("2D Peak Width Guess", "Input your estimate for the width of the 2D Peak")

    for peak in all_peaks:
        peaks[peak] = []
        areas[peak] = []

        messagebox.showinfo("Type of Peak", f"This is the analysis of the {peak} peak.")
        root = Tk()
        root.withdraw()  # This hides the root window, so only the dialog will be shown

        small_range, large_range = fixed_ranges.get(peak, (0.0, 0.0))

        bounds_L_low = np.array([0, small_range, 0])
        bounds_L_high = np.array([np.inf, large_range, np.inf])

        fixed_initial_guesses = {
            'D': {
                "Lorentzian": [D_area_guess, 1350, D_width_guess]
            },
            'G': {
                "Lorentzian": [G_area_guess, 1580, G_width_guess]
            },
            '2D': {
                "Lorentzian": [TwoD_area_guess, 2700, TwoD_width_guess]
            }
        }

        initial_guesses_L = fixed_initial_guesses[peak]["Lorentzian"]

        for i in range(len(x_coordinates)):
            try:
                mask_data_x_s = np.ma.masked_array(shifts, shifts <= small_range)
                smallest = np.ma.argmin(mask_data_x_s)

                mask_data_x_l = np.ma.masked_array(shifts, shifts >= large_range)
                largest = np.ma.argmax(mask_data_x_l)

                shifts_peak = shifts[smallest:largest+1]
                intensities_peak = intensities[i, smallest:largest+1]

                y_first = np.mean(intensities_peak[-10:])
                y_last = np.mean(intensities_peak[:10])
                x_first = np.mean(shifts_peak[-10:])
                x_last = np.mean(shifts_peak[:10])
                const = (intensities_peak.max() - intensities_peak.min()) / 35

                popt, pcov = optimize.curve_fit(fit_L, shifts_peak, intensities_peak, p0=initial_guesses_L, bounds=(bounds_L_low, bounds_L_high), maxfev=10000)
                
                A_err = np.sqrt(pcov[0][0])  # Uncertainty in A, sqaure root of variance
            #

                areas[peak].append(popt[0])
                peaks[peak].append(popt[1])

                if peak == 'D':
                    D_area = popt[0]
                    D_area_err = A_err
                elif peak == 'G':
                    G_area = popt[0]
                    G_area_err = A_err
                    
                    # Calculate A(D)/A(G) and its uncertainty
                    if 'D' in areas:
                        ratio = D_area / G_area # Ratio areas
                        ratio_err = ratio * np.sqrt((D_area_err/D_area)**2 + (G_area_err/G_area)**2) # Error propagation formula # type: ignore
                        AD_AG_ratio.append(ratio) # append ratio to variable
                        AD_AG_ratio_err.append(ratio_err) # append +/- calculation of error (what goes after  +/-)


            except RuntimeError as e:
                print(f"An error occurred at index {i} for peak {peak}: {e}")
                dummy_area, dummy_peak = {
                    'G': (100, 100),
                    'D': (0.1, 100),
                    '2D': (0.1, 100),
                }.get(peak, (1e6, 1e6))  # default dummy values if peak is not 'G', 'D', or '2D'
                areas[peak].append(dummy_area)
                peaks[peak].append(dummy_peak)

            if i == peak_index:
                plt.figure()
                plt.plot(shifts_peak, intensities_peak, 'x')
                plt.plot(shifts_peak, fit_L(shifts_peak, *popt), color='black')
                plt.title(f'Sample ID: {sample_id}')
                plt.show()

    if 'G' in peaks:
        # 3D plot of pos(G)
        min_z = min(peaks['G']) # meaning the min vals of these peaks 
        max_z = max(peaks['G'])
        while True:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            cmap = plt.get_cmap('viridis')
            sc = ax.scatter(x_coordinates, y_coordinates, peaks['G'], c=peaks['G'], cmap=cmap, marker='o', s=5000/len(x_coordinates), vmin=min_z, vmax=max_z, alpha=0.7) # type: ignore
            cbar = fig.colorbar(sc)
            cbar.set_label('pos(G) / cm$^{-1}$')
            ax.set_xlabel('x-position / μm')
            ax.set_ylabel('y-position / μm')
            ax.set_zlabel('pos(G) / cm$^{-1}$') # type: ignore
            ax.set_zlim3d(min_z, max_z) # type: ignore
            ax.grid(visible=False)
            plt.title(f'pos(G) - Sample ID: {sample_id}')
            plt.show()

            change_bounds = simpledialog.askstring("Axes", 'Would you like to change the axes? Input "Yes" or "No":')
            if change_bounds == 'Yes':
                min_z = simpledialog.askfloat("Minimum z", "What is the smallest value of the z-axis?")
                max_z = simpledialog.askfloat("Maximum z", "What is the largest value of the z-axis?")
            else:
                break

        # Calculate the distance between two adjacent data points
        if len(x_coordinates) > 1 and len(y_coordinates) > 1:
            distance = np.sqrt((x_coordinates[1] - x_coordinates[0])**2 + (y_coordinates[1] - y_coordinates[0])**2)
            s = (distance * 0.8)**2  # Calculate the marker size as a fraction of the distance
        else:
            s = 20  # Default marker size if not enough points to calculate distance

        # 2D plot of pos(G)
        xi, yi = np.meshgrid(np.linspace(min(x_coordinates), max(x_coordinates), 500),
                             np.linspace(min(y_coordinates), max(y_coordinates), 500))
        zi = griddata((x_coordinates.flatten(), y_coordinates.flatten()), np.array(peaks['G']), (xi, yi), method='nearest')

        
        # Apply a Gaussian filter
        while True:
            smoothed_zi = gaussian_filter(zi, sigma=2)

            # Define levels
            contourf_levels = np.linspace(min_z, max_z, 400)  # This creates 400 levels for smoother colormap # type: ignore

            # Create filled contours
            plt.contourf(xi, yi, smoothed_zi, contourf_levels, cmap='viridis')

            # Add colorbar
            plt.colorbar(label='pos(G) / cm$^{-1}$')

            sc = plt.scatter(x_coordinates, y_coordinates, c=peaks['G'], s=s, alpha=0)  # Invisible scatter for mplcursors

            plt.gca().set_aspect('equal')
            plt.xlim([min(x_coordinates), max(x_coordinates)])
            plt.ylim([min(y_coordinates), max(y_coordinates)])
            
            plt.xlabel('x-position / μm')
            plt.ylabel('y-position / μm')
            plt.title(f'pos(G) - Sample ID: {sample_id}')

        # Replica of original fitting # Replica of original fitting 
        # Replica of original fitting # Replica of original fitting
        

            def fit_L(x, A, P, W, y_first, y_last, x_first, x_last, const):
                backline = (y_last - y_first) / (x_last - x_first) * (x - x_first) + y_first - const
                return A * W / ((np.pi) * ((x - P)**2 + W**2)) + backline


            def fit_peak(shifts, intensities, peak, bounds_L_low, bounds_L_high, initial_guesses_L):
                mask_data_x_s = np.ma.masked_array(shifts, shifts <= bounds_L_low[1])
                smallest = np.ma.argmin(mask_data_x_s)
                mask_data_x_l = np.ma.masked_array(shifts, shifts >= bounds_L_high[1])
                largest = np.ma.argmax(mask_data_x_l)

                shifts_peak = shifts[smallest:largest+1]
                intensities_peak = intensities[smallest:largest+1]

                # Recalculate background parameters
                y_first = np.mean(intensities_peak[-10:])
                y_last = np.mean(intensities_peak[:10])
                x_first = np.mean(shifts_peak[-10:])
                x_last = np.mean(shifts_peak[:10])
                const = (intensities_peak.max() - intensities_peak.min()) / 35

                # Ensure initial guesses are within bounds
                initial_guesses_L = np.clip(initial_guesses_L, bounds_L_low, bounds_L_high)

                try:
                    popt, _ = optimize.curve_fit(
                        lambda x, A, P, W: fit_L(x, A, P, W, y_first, y_last, x_first, x_last, const),
                        shifts_peak, intensities_peak, p0=initial_guesses_L,
                        bounds=(bounds_L_low, bounds_L_high), maxfev=10000
                    )
                    fitted_curve = fit_L(shifts_peak, *popt, y_first, y_last, x_first, x_last, const) # type: ignore
                    return shifts_peak, fitted_curve, popt

                except RuntimeError as e:
                    print(f"An error occurred for peak {peak}: {e}")
                    return shifts_peak, None, None

            # Replica of original fitting 
            # Replica of original fitting

            def on_click(event):
                ix, iy = event.xdata, event.ydata
                dist = (x_coordinates - ix)**2 + (y_coordinates - iy)**2
                index = np.argmin(dist)

                plt.figure(figsize=(10, 6))
                plt.plot(shifts, intensities[index], label='Raw data')

                max_intensity = np.max(intensities[index])

                # Add fitting curves for each peak
                for peak in all_peaks:
                    small_range, large_range = fixed_ranges[peak]

                    # Set bounds and initial guesses based on the peak type
                    bounds_L_low = np.array([0, small_range, 0])
                    bounds_L_high = np.array([np.inf, large_range, np.inf])
                    initial_guesses_L = fixed_initial_guesses[peak]["Lorentzian"]

                    # Fit the peak and plot
                    shifts_peak, fitted_curve, popt = fit_peak(
                        shifts, intensities[index], peak,
                        bounds_L_low, bounds_L_high, initial_guesses_L
                    )

                    if fitted_curve is not None:
                        plt.plot(shifts_peak, fitted_curve, '--', label=f'{peak} fit')

                        # Print details for debugging
                        print(f"Peak: {peak}")
                        print(f"A: {popt[0]}, P: {popt[1]}, W: {popt[2]}") # type: ignore
                        print(f"Max of fitted curve: {np.max(fitted_curve)}")
                        print(f"Max of raw data in this range: {np.max(intensities[index])}")

                # Adjust plot limits to make sure all data is visible
                plt.xlim(shifts.min(), shifts.max())
                plt.ylim(0, max_intensity)
                
                plt.title(f'Spectrum at x: {x_coordinates[index]:.2f}, y: {y_coordinates[index]:.2f} - Sample ID: {sample_id}')
                plt.xlabel('Raman Shift (cm-1)')
                plt.ylabel('Intensity')
                plt.legend()
                plt.show()

            plt.gcf().canvas.mpl_connect('button_press_event', on_click)
            plt.show()

            

            change_bounds = simpledialog.askstring("Axes", 'Would you like to change the axes? Input "Yes" or "No":')
            if change_bounds == 'Yes':
                    min_z = simpledialog.askfloat("Minimum z", "What is the smallest value of the z-axis?")
                    max_z = simpledialog.askfloat("Maximum z", "What is the largest value of the z-axis?")
            else:
                break 

    if '2D' in peaks:
            
            #pos(2D) against pos(G)
            def theoliney(x):
                y = 191/63*(x-1585) + 2750
                return y

            # Generate the theoretical line data
            theo_x = np.linspace(1580, 1595, 1000)
            theo_y = theoliney(theo_x)

            # Create a figure with 2 subplots
            plt.figure(figsize=(12, 6))

            # Full scale subplot
            plt.subplot(1, 2, 1)
            plt.scatter(peaks['G'], peaks['2D'], label='Data points')
            plt.plot(theo_x, theo_y, linestyle='--', color='red', label='Theoretical Line')
            plt.xlabel('pos(G) / cm$^{-1}$')
            plt.ylabel('pos(2D) / cm$^{-1}$')
            plt.title(f'pos(2D) against pos(G) - Sample ID: {sample_id}')
            plt.legend()

            # Constant scale subplot
            plt.subplot(1, 2, 2)
            plt.scatter(peaks['G'], peaks['2D'], label='Data points')
            plt.plot(theo_x, theo_y, linestyle='--', color='red', label='Theoretical Line')
            plt.xlim(1575, 1605)
            plt.ylim(2720, 2800)
            plt.xlabel('pos(G) / cm$^{-1}$')
            plt.ylabel('pos(2D) / cm$^{-1}$')
            plt.title(f'pos(2D) against pos(G) - Sample ID: {sample_id} (Constant Scale)')
            plt.legend()

            # Show the plots
            plt.tight_layout()
            plt.show()

            # 3D plot of pos(2D)
            min_z = min(peaks['2D'])
            max_z = max(peaks['2D'])


            while True:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                cmap = plt.colormaps.get_cmap('plasma')
                sc = ax.scatter(x_coordinates, y_coordinates, peaks['2D'], c=peaks['2D'], cmap=cmap, marker='o', s=5000/len(x_coordinates), vmin=min_z, vmax=max_z, alpha=0.7) # type: ignore
                cbar = fig.colorbar(sc)
                cbar.set_label('pos(2D) / cm$^{-1}$')
                ax.set_xlabel('x-position / μm')
                ax.set_ylabel('y-position / μm')
                ax.set_zlabel('pos(2D) / cm$^{-1}$') # type: ignore
                ax.set_zlim3d(min_z, max_z) # type: ignore
                ax.grid(visible=False)
                plt.title(f'pos(2D) - Sample ID: {sample_id}')
                plt.show()

                change_bounds = simpledialog.askstring("Axes", 'Would you like to change the axes? Input "Yes" or "No":')
                if change_bounds == 'Yes':
                    min_z = simpledialog.askfloat("Minimum z","What is the smallest value of the z-axis?")
                    max_z = simpledialog.askfloat("Maximum z","What is the largest value of the z-axis?")
                else:
                    break
            # calculate the distance between two adjacent data points
            distance = np.sqrt((x_coordinates[1] - x_coordinates[0])**2 + (y_coordinates[1] - y_coordinates[0])**2)

            # calculate the marker size as a fraction of the distance
            s = (distance * 0.8)**2
            # 2D plot of pos(2D)
            xi, yi = np.meshgrid(np.linspace(min(x_coordinates), max(x_coordinates), 500),
                                np.linspace(min(y_coordinates), max(y_coordinates), 500))
            zi = griddata((x_coordinates.flatten(), y_coordinates.flatten()), np.array(peaks['2D']), (xi, yi), method='nearest')

            while True:
                smoothed_zi = gaussian_filter(zi, sigma=2)
                levels = np.linspace(min_z, max_z, 400) # type: ignore

                plt.contourf(xi, yi, smoothed_zi, levels, cmap='plasma')
                plt.colorbar(label='pos(2D) / cm$^{-1}$')
                plt.scatter(x_coordinates, y_coordinates, c=peaks['2D'], cmap='Greys', marker='s', s=0, vmin=min_z, vmax=max_z, edgecolors='none')
                plt.gca().set_aspect('equal')
                # Set the axes limits to fit the data range
                plt.xlim([min(x_coordinates), max(x_coordinates)])
                plt.ylim([min(y_coordinates), max(y_coordinates)])
                plt.xlabel('x-position / μm')
                plt.ylabel('y-position / μm')
                plt.title(f'pos(2D) - Sample ID: {sample_id}')


                def fit_L(x, A, P, W, y_first, y_last, x_first, x_last, const):
                    backline = (y_last - y_first) / (x_last - x_first) * (x - x_first) + y_first - const
                    return A * W / ((np.pi) * ((x - P)**2 + W**2)) + backline

                def fit_peak(shifts, intensities, peak, bounds_L_low, bounds_L_high, initial_guesses_L):
                    mask_data_x_s = np.ma.masked_array(shifts, shifts <= bounds_L_low[1])
                    smallest = np.ma.argmin(mask_data_x_s)
                    mask_data_x_l = np.ma.masked_array(shifts, shifts >= bounds_L_high[1])
                    largest = np.ma.argmax(mask_data_x_l)

                    shifts_peak = shifts[smallest:largest+1]
                    intensities_peak = intensities[smallest:largest+1]

                    # Recalculate background parameters
                    y_first = np.mean(intensities_peak[-10:])
                    y_last = np.mean(intensities_peak[:10])
                    x_first = np.mean(shifts_peak[-10:])
                    x_last = np.mean(shifts_peak[:10])
                    const = (intensities_peak.max() - intensities_peak.min()) / 35

                    # Ensure initial guesses are within bounds
                    initial_guesses_L = np.clip(initial_guesses_L, bounds_L_low, bounds_L_high)

                    try:
                        popt, _ = optimize.curve_fit(
                            lambda x, A, P, W: fit_L(x, A, P, W, y_first, y_last, x_first, x_last, const),
                            shifts_peak, intensities_peak, p0=initial_guesses_L,
                            bounds=(bounds_L_low, bounds_L_high), maxfev=10000
                        )
                        fitted_curve = fit_L(shifts_peak, *popt, y_first, y_last, x_first, x_last, const) # type: ignore
                        return shifts_peak, fitted_curve, popt

                    except RuntimeError as e:
                        print(f"An error occurred for peak {peak}: {e}")
                        return shifts_peak, None, None

                def on_click(event):
                    ix, iy = event.xdata, event.ydata
                    dist = (x_coordinates - ix)**2 + (y_coordinates - iy)**2
                    index = np.argmin(dist)

                    plt.figure(figsize=(10, 6))
                    plt.plot(shifts, intensities[index], label='Raw data')

                    max_intensity = np.max(intensities[index])

                    # Add fitting curves for each peak
                    for peak in all_peaks:
                        small_range, large_range = fixed_ranges[peak]

                        # Set bounds and initial guesses based on the peak type
                        bounds_L_low = np.array([0, small_range, 0])
                        bounds_L_high = np.array([np.inf, large_range, np.inf])
                        initial_guesses_L = fixed_initial_guesses[peak]["Lorentzian"]

                        # Fit the peak and plot
                        shifts_peak, fitted_curve, popt = fit_peak(
                            shifts, intensities[index], peak,
                            bounds_L_low, bounds_L_high, initial_guesses_L
                        )

                        if fitted_curve is not None:
                            plt.plot(shifts_peak, fitted_curve, '--', label=f'{peak} fit')

                            # Print details for debugging
                            print(f"Peak: {peak}")
                            print(f"A: {popt[0]}, P: {popt[1]}, W: {popt[2]}") # type: ignore
                            print(f"Max of fitted curve: {np.max(fitted_curve)}")
                            print(f"Max of raw data in this range: {np.max(intensities[index])}") 

                    # Adjust plot limits to make sure all data is visible
                    plt.xlim(shifts.min(), shifts.max())
                    plt.ylim(0, max_intensity)
                    
                    plt.title(f'Spectrum at x: {x_coordinates[index]:.2f}, y: {y_coordinates[index]:.2f} - Sample ID: {sample_id}')
                    plt.xlabel('Raman Shift (cm-1)')
                    plt.ylabel('Intensity')
                    plt.legend()
                    plt.show()

                plt.gcf().canvas.mpl_connect('button_press_event', on_click)
                plt.show()

                change_bounds = simpledialog.askstring("Axes", 'Would you like to change the axes? Input "Yes" or "No":')
                if change_bounds == 'Yes':
                        min_z = simpledialog.askfloat("Minimum z", "What is the smallest value of the z-axis?")
                        max_z = simpledialog.askfloat("Maximum z", "What is the largest value of the z-axis?")
                else:
                    break 
                
    if 'G' in peaks and 'D' in peaks:
            ratio_area = np.array(areas['D'])/np.array(areas['G'])
            min_z = min(ratio_area)
            max_z = max(ratio_area)
            while True:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                cmap = plt.colormaps.get_cmap('Greys')
                sc = ax.scatter(x_coordinates, y_coordinates, ratio_area, c=ratio_area, cmap='viridis', marker='o', s=5000/len(x_coordinates), vmin=min_z, vmax=max_z, alpha=0.7) # type: ignore / s=5000/len(x_coordinates)
                cbar = fig.colorbar(sc)
                cbar.set_label('A(D)/A(G)')
                ax.set_xlabel('x-position / μm')
                ax.set_ylabel('y-position / μm')
                ax.set_zlabel('A(D)/A(G)') # type: ignore
                ax.set_zlim3d(min_z, max_z) # type: ignore
                ax.grid(visible=False)
                plt.title(f'A(D)/A(G) - Sample ID: {sample_id}')
                plt.show()

                change_bounds = simpledialog.askstring("Axes", 'Would you like to change the axes? Input "Yes" or "No":')
                if change_bounds == 'Yes':
                    min_z = simpledialog.askfloat("Minimum z","What is the smallest value of the z-axis?")
                    max_z = simpledialog.askfloat("Maximum z","What is the largest value of the z-axis?")
                else:
                    break

            # A(D)/A(G)
            xi, yi = np.meshgrid(np.linspace(min(x_coordinates), max(x_coordinates), 500),
                                np.linspace(min(y_coordinates), max(y_coordinates), 500))
            zi = griddata((x_coordinates.flatten(), y_coordinates.flatten()), ratio_area, (xi, yi), method='nearest')

            while True:
            # apply a Gaussian filter
                smoothed_zi = gaussian_filter(zi, sigma=2)

                levels = np.linspace(min_z, max_z, 400) # type: ignore

                plt.contourf(xi, yi, smoothed_zi, levels, cmap='viridis')
                plt.colorbar(label='A(D)/A(G)')
                plt.scatter(x_coordinates, y_coordinates, c=ratio_area, cmap='Greys', marker='s', s=0, vmin=min_z, vmax=max_z, edgecolors='none')
                plt.gca().set_aspect('equal')
                # Set the axes limits to fit the data range
                plt.xlim([min(x_coordinates), max(x_coordinates)])
                plt.ylim([min(y_coordinates), max(y_coordinates)])
                plt.xlabel('x-position / μm')
                plt.ylabel('y-position / μm')
                plt.title(f'A(D)/A(G) - Sample ID: {sample_id}')

                def fit_L(x, A, P, W, y_first, y_last, x_first, x_last, const):
                    backline = (y_last - y_first) / (x_last - x_first) * (x - x_first) + y_first - const
                    return A * W / ((np.pi) * ((x - P)**2 + W**2)) + backline
                
                #Parameters:

                #x: The independent variable (Raman shift).
                #A: The amplitude of the Lorentzian peak.
                #P: The position of the Lorentzian peak.
                #W: The width of the Lorentzian peak.
                #y_first: The intensity at the start of the background range.
                #y_last: The intensity at the end of the background range.
                #x_first: The shift at the start of the background range.
                #x_last: The shift at the end of the background range.
                #const: A constant to adjust the background level.

                def fit_peak(shifts, intensities, peak, bounds_L_low, bounds_L_high, initial_guesses_L):
                    mask_data_x_s = np.ma.masked_array(shifts, shifts <= bounds_L_low[1])
                    smallest = np.ma.argmin(mask_data_x_s)
                    mask_data_x_l = np.ma.masked_array(shifts, shifts >= bounds_L_high[1])
                    largest = np.ma.argmax(mask_data_x_l)

                    shifts_peak = shifts[smallest:largest+1]
                    intensities_peak = intensities[smallest:largest+1]

                    # Recalculate background parameters
                    y_first = np.mean(intensities_peak[-10:])
                    y_last = np.mean(intensities_peak[:10])
                    x_first = np.mean(shifts_peak[-10:])
                    x_last = np.mean(shifts_peak[:10])
                    const = (intensities_peak.max() - intensities_peak.min()) / 35

                    # Ensure initial guesses are within bounds
                    initial_guesses_L = np.clip(initial_guesses_L, bounds_L_low, bounds_L_high)

                    try:
                        popt, _ = optimize.curve_fit(
                            lambda x, A, P, W: fit_L(x, A, P, W, y_first, y_last, x_first, x_last, const),
                            shifts_peak, intensities_peak, p0=initial_guesses_L,
                            bounds=(bounds_L_low, bounds_L_high), maxfev=10000
                        )
                        fitted_curve = fit_L(shifts_peak, *popt, y_first, y_last, x_first, x_last, const) # type: ignore
                        return shifts_peak, fitted_curve, popt

                    except RuntimeError as e:
                        print(f"An error occurred for peak {peak}: {e}")
                        return shifts_peak, None, None
                    
                    #Parameters:

                    #shifts: Array of Raman shifts.
                    #intensities: Array of intensities corresponding to the Raman shifts.
                    #peak: Name of the peak being fitted.
                    #bounds_L_low: Lower bounds for the fitting parameters [A, P, W].
                    #bounds_L_high: Upper bounds for the fitting parameters [A, P, W].
                    #initial_guesses_L: Initial guesses for the fitting parameters [A, P, W].

                    #Steps:

                    #Masks the shifts array to find the range for fitting based on bounds_L_low and bounds_L_high.
                    #Extracts the relevant range of shifts and intensities for the peak.
                    #Calculates the background parameters (y_first, y_last, x_first, x_last, const).
                    #Clips the initial guesses to ensure they are within the specified bounds.
                    #Tries to fit the Lorentzian model to the data using optimize.curve_fit.
                    #Returns the fitted peak data (shifts_peak, fitted_curve, popt).
                    #Returns:

                    #shifts_peak: The range of Raman shifts used for fitting.
                    #fitted_curve: The fitted Lorentzian curve plus background.
                    #popt: The optimized parameters of the Lorentzian fit.

                def on_click(event):
                    ix, iy = event.xdata, event.ydata
                    dist = (x_coordinates - ix)**2 + (y_coordinates - iy)**2
                    index = np.argmin(dist)

                    plt.figure(figsize=(10, 6))
                    plt.plot(shifts, intensities[index], label='Raw data')

                    max_intensity = np.max(intensities[index])

                    # Add fitting curves for each peak
                    for peak in all_peaks:
                        small_range, large_range = fixed_ranges[peak]

                        # Set bounds and initial guesses based on the peak type
                        bounds_L_low = np.array([0, small_range, 0])
                        bounds_L_high = np.array([np.inf, large_range, np.inf])
                        initial_guesses_L = fixed_initial_guesses[peak]["Lorentzian"]

                        # Fit the peak and plot
                        shifts_peak, fitted_curve, popt = fit_peak(
                            shifts, intensities[index], peak,
                            bounds_L_low, bounds_L_high, initial_guesses_L
                        )

                        if fitted_curve is not None:
                            plt.plot(shifts_peak, fitted_curve, '--', label=f'{peak} fit')

                            # Print details for debugging
                            print(f"Peak: {peak}")
                            print(f"A: {popt[0]}, P: {popt[1]}, W: {popt[2]}") # type: ignore
                            print(f"Max of fitted curve: {np.max(fitted_curve)}")
                            print(f"Max of raw data in this range: {np.max(intensities[index])}")

                    # Adjust plot limits to make sure all data is visible
                    plt.xlim(shifts.min(), shifts.max())
                    plt.ylim(0, max_intensity)
                    plt.title(f'Spectrum at x: {x_coordinates[index]:.2f}, y: {y_coordinates[index]:.2f} - Sample ID: {sample_id}')
                    plt.xlabel('Raman Shift (cm-1)')
                    plt.ylabel('Intensity')
                    plt.legend()
                    plt.show() 

                    #Retrieves the x and y coordinates of the click event.
                    #Calculates the distance from the click to each data point and finds the closest point.
                    #Plots the raw data spectrum at the selected location.
                    #Iterates over each peak type in all_peaks, setting the bounds and initial guesses for fitting.
                    #Calls fit_peak to fit the Lorentzian model to the data for each peak.
                    #Plots the fitted curve if the fitting was successful.
                    #Adjusts the plot limits and displays the plot with labels and a legend.

                    if 'D' in peaks and 'G' in peaks:
                        ratio = AD_AG_ratio[index]
                        ratio_err = AD_AG_ratio_err[index]
                        messagebox.showinfo("A(D)/A(G) Ratio", f"A(D)/A(G) ratio at this point: {ratio:.2f} ± {ratio_err:.2f}")   
                        

                        # show on msg box the ratio with its respective error, determined by the place we click (gets stored in "index")


                plt.gcf().canvas.mpl_connect('button_press_event', on_click)
                plt.show()

                

                change_bounds = simpledialog.askstring("Axes", 'Would you like to change the axes? Input "Yes" or "No":')
                if change_bounds == 'Yes':
                        min_z = simpledialog.askfloat("Minimum z", "What is the smallest value of the z-axis?")
                        max_z = simpledialog.askfloat("Maximum z", "What is the largest value of the z-axis?")
                else:
                    break


            # Calculate quartiles and IQR for outlier detection
            q1 = np.percentile(ratio_area, 25)
            q3 = np.percentile(ratio_area, 75)
            iqr = q3 - q1

            # Define bounds and filter outliers
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            filtered_data = [x for x in ratio_area if lower_bound <= x <= upper_bound]

            # Create histogram and calculate bin centers
            hist, bin_edges = np.histogram(filtered_data, bins=20)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Fit single Gaussian
            popt_single, pcov_single = curve_fit(gaussian, bin_centers, hist, 
                                                p0=[hist.max(), np.mean(filtered_data), np.std(filtered_data)])
            perr_single = np.sqrt(np.diag(pcov_single))

            # Fit double Gaussian
            initial_guess = [hist.max()/2, np.mean(filtered_data)-np.std(filtered_data), np.std(filtered_data)/2,
                            hist.max()/2, np.mean(filtered_data)+np.std(filtered_data), np.std(filtered_data)/2]
            popt_double, pcov_double = curve_fit(double_gaussian, bin_centers, hist, 
                                                p0=initial_guess, maxfev=100000)
            perr_double = np.sqrt(np.diag(pcov_double))

            # Calculate BIC for single and double Gaussian models
            n = len(hist)
            k_single = 3  # number of parameters for single Gaussian
            k_double = 6  # number of parameters for double Gaussian
            residuals_single = hist - gaussian(bin_centers, *popt_single)
            residuals_double = hist - double_gaussian(bin_centers, *popt_double)
            bic_single = n * np.log(np.sum(residuals_single**2) / n) + k_single * np.log(n)
            bic_double = n * np.log(np.sum(residuals_double**2) / n) + k_double * np.log(n)

            # Calculate ΔBIC and determine significance
            delta_bic = bic_double - bic_single
            if delta_bic < 2:
                significance = "Little to no evidence against the model with the lower BIC."
                probability = 1 / (1 + np.exp(delta_bic / 2))
            elif 2 <= delta_bic < 6:
                significance = "Positive evidence against the model with the higher BIC."
                probability = 1 / (1 + np.exp(delta_bic / 2))
            elif 6 <= delta_bic < 10:
                significance = "Strong evidence against the model with the higher BIC."
                probability = 1 / (1 + np.exp(delta_bic / 2))
            else:
                significance = "Very strong evidence against the model with the higher BIC."
                probability = 1 / (1 + np.exp(delta_bic / 2))

            # Plot results
            plt.figure(figsize=(12, 6))
            plt.hist(filtered_data, bins=20, edgecolor='black', alpha=0.7, label='Data')
            plt.plot(bin_centers, gaussian(bin_centers, *popt_single), 'r-', label='Single Gaussian Fit')
            plt.plot(bin_centers, double_gaussian(bin_centers, *popt_double), 'g-', label='Double Gaussian Fit')

            # Plot individual Gaussians from the double Gaussian fit
            gaussian1 = gaussian(bin_centers, popt_double[0], popt_double[1], popt_double[2])
            gaussian2 = gaussian(bin_centers, popt_double[3], popt_double[4], popt_double[5])
            plt.plot(bin_centers, gaussian1, 'b--', label='Gaussian 1')
            plt.plot(bin_centers, gaussian2, 'm--', label='Gaussian 2')

            # Set axis labels and title
            plt.xlabel('A(D)/A(G)')
            plt.ylabel('Frequency')
            plt.title(f'A(D)/A(G) Gaussian Fits Comparison - Sample ID: {sample_id}')
            plt.legend()
            plt.show()

            # Print fitting results and BIC comparison
            print(f"Single Gaussian BIC: {bic_single:.2f}")
            print(f"Double Gaussian BIC: {bic_double:.2f}")
            print(f"ΔBIC: {delta_bic:.2f} ({significance})")
            print(f"Probability that Single Gaussian is better: {probability:.2%}")

            # Print fitting results and include mean and SEM
            ratio_type = 'A(D)/A(G)'
            mean_ratio = np.mean(filtered_data) #Filt data is AD/AG, takes the info from line540 (ratio_area)
            sem_ratio = np.std(filtered_data, ddof=1) / np.sqrt(len(filtered_data)) # ddof is degrees of freedom, value is 1 to calc sample stand deviation
            print(f"\nMean {ratio_type}: {mean_ratio:.2f} ± {sem_ratio:.2f}")

            print("Single Gaussian Fit Parameters:")
            print(f"Amplitude: {popt_single[0]:.2f} ± {perr_single[0]:.2f}")
            print(f"Centre: {popt_single[1]:.2f} ± {perr_single[1]:.2f}")
            print(f"Width: {popt_single[2]:.2f} ± {perr_single[2]:.2f}")
            print(f"BIC: {bic_single:.2f}")

            print("\nDouble Gaussian Fit Parameters:")
            print(f"Amplitude 1: {popt_double[0]:.2f} ± {perr_double[0]:.2f}")
            print(f"Centre 1: {popt_double[1]:.2f} ± {perr_double[1]:.2f}")
            print(f"Width 1: {popt_double[2]:.2f} ± {perr_double[2]:.2f}")
            print(f"Amplitude 2: {popt_double[3]:.2f} ± {perr_double[3]:.2f}")
            print(f"Centre 2: {popt_double[4]:.2f} ± {perr_double[4]:.2f}")
            print(f"Width 2: {popt_double[5]:.2f} ± {perr_double[5]:.2f}")
            print(f"BIC: {bic_double:.2f}")


            #For the single Gaussian fit:

                #popt_single[0] and perr_single[0]:

                #Index 0 corresponds to the Amplitude parameter.
                #This is the first parameter in the Gaussian function definition.


                #popt_single[1] and perr_single[1]:

                #Index 1 corresponds to the Centre parameter.
                #This is the second parameter in the Gaussian function definition.


                #popt_single[2] and perr_single[2]:

                #Index 2 corresponds to the Width parameter.
                #This is the third parameter in the Gaussian function definition.

                #The order of these indices (0, 1, 2) matches the order of parameters in the gaussian function we defined earlier: def gaussian(x, amp, centre, width).


            # Double Gaussian Fit

                #popt_double[0] and perr_double[0]: Amplitude of the first Gaussian
                #popt_double[1] and perr_double[1]: Centre of the first Gaussian
                #popt_double[2] and perr_double[2]: Width of the first Gaussian
                #popt_double[3] and perr_double[3]: Amplitude of the second Gaussian
                #popt_double[4] and perr_double[4]: Centre of the second Gaussian
                #popt_double[5] and perr_double[5]: Width of the second Gaussian

                #These indices (0 to 5) correspond to the order of parameters in the double_gaussian function: def double_gaussian(x, amp1, centre1, width1, amp2, centre2, width2).
                #The popt_* arrays contain the optimal parameter values found by the curve fitting process, while the perr_* arrays contain the standard errors of these parameters.
                #The :.2f in each f-string formats the numbers to display with 2 decimal places.
                #Lastly, bic_single and bic_double are scalar values (not arrays), so they don't use indexing. They represent the Bayesian Information Criterion for each model, which is calculated separately.



    if 'G' in peaks and '2D' in peaks:
        ratio_area = np.array(areas['2D'])/np.array(areas['G'])
        min_z = min(ratio_area)
        max_z = max(ratio_area)  
        while True:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            cmap = plt.colormaps.get_cmap('Greys')
            sc = ax.scatter(x_coordinates, y_coordinates, ratio_area, c=ratio_area, cmap='plasma', marker='o', s=5000/len(x_coordinates), vmin=min_z, vmax=max_z, alpha=0.7) # type: ignore
            cbar = fig.colorbar(sc)
            cbar.set_label('A(2D)/A(G)')
            ax.set_xlabel('x-position / μm')
            ax.set_ylabel('y-position / μm')
            ax.set_zlabel('A(2D)/A(G)')# type: ignore
            ax.set_zlim3d(min_z, max_z)# type: ignore
            ax.grid(visible=False)
            plt.title(f'A(2D)/A(G) - Sample ID: {sample_id}')
            plt.show()

            change_bounds = simpledialog.askstring("Axes", 'Would you like to change the axes? Input "Yes" or "No":')
            if change_bounds == 'Yes':
                min_z = simpledialog.askfloat("Minimum z","What is the smallest value of the z-axis?")
                max_z = simpledialog.askfloat("Maximum z","What is the largest value of the z-axis?")
            else:
                break

        # A(2D)/A(G)
        xi, yi = np.meshgrid(np.linspace(min(x_coordinates), max(x_coordinates), 500),
                             np.linspace(min(y_coordinates), max(y_coordinates), 500))
        zi = griddata((x_coordinates.flatten(), y_coordinates.flatten()), ratio_area, (xi, yi), method='nearest')

        while True:
        # apply a Gaussian filter
            smoothed_zi = gaussian_filter(zi, sigma=2)

            levels = np.linspace(min_z, max_z, 400) # type: ignore

            plt.contourf(xi, yi, smoothed_zi, levels, cmap='plasma')
            plt.colorbar(label='A(2D)/A(G)')
            plt.scatter(x_coordinates, y_coordinates, c=ratio_area, cmap='Greys', marker='s', s=0, vmin=min_z, vmax=max_z, edgecolors='none') # Usually is s = s, trying = 0 to make dots dissapear
            plt.gca().set_aspect('equal')
            # Set the axes limits to fit the data range
            plt.xlim([min(x_coordinates), max(x_coordinates)])
            plt.ylim([min(y_coordinates), max(y_coordinates)])
            plt.xlabel('x-position / μm')
            plt.ylabel('y-position / μm')
            plt.title(f'A(2D)/A(G) - Sample ID: {sample_id}')

            def fit_L(x, A, P, W, y_first, y_last, x_first, x_last, const):
                backline = (y_last - y_first) / (x_last - x_first) * (x - x_first) + y_first - const
                return A * W / ((np.pi) * ((x - P)**2 + W**2)) + backline

            def fit_peak(shifts, intensities, peak, bounds_L_low, bounds_L_high, initial_guesses_L):
                mask_data_x_s = np.ma.masked_array(shifts, shifts <= bounds_L_low[1])
                smallest = np.ma.argmin(mask_data_x_s)
                mask_data_x_l = np.ma.masked_array(shifts, shifts >= bounds_L_high[1])
                largest = np.ma.argmax(mask_data_x_l)

                shifts_peak = shifts[smallest:largest+1] 
                intensities_peak = intensities[smallest:largest+1]

                # Recalculate background parameters
                y_first = np.mean(intensities_peak[-10:])
                y_last = np.mean(intensities_peak[:10])
                x_first = np.mean(shifts_peak[-10:])
                x_last = np.mean(shifts_peak[:10])
                const = (intensities_peak.max() - intensities_peak.min()) / 35

                # Ensure initial guesses are within bounds
                initial_guesses_L = np.clip(initial_guesses_L, bounds_L_low, bounds_L_high)

                try:
                    popt, _ = optimize.curve_fit(
                        lambda x, A, P, W: fit_L(x, A, P, W, y_first, y_last, x_first, x_last, const),
                        shifts_peak, intensities_peak, p0=initial_guesses_L,
                        bounds=(bounds_L_low, bounds_L_high), maxfev=10000
                    )
                    fitted_curve = fit_L(shifts_peak, *popt, y_first, y_last, x_first, x_last, const) # type: ignore
                    return shifts_peak, fitted_curve, popt

                except RuntimeError as e:
                    print(f"An error occurred for peak {peak}: {e}")
                    return shifts_peak, None, None

            def on_click(event):
                ix, iy = event.xdata, event.ydata
                dist = (x_coordinates - ix)**2 + (y_coordinates - iy)**2
                index = np.argmin(dist)

                plt.figure(figsize=(10, 6))
                plt.plot(shifts, intensities[index], label='Raw data')

                max_intensity = np.max(intensities[index])

                # Add fitting curves for each peak
                for peak in all_peaks:
                    small_range, large_range = fixed_ranges[peak]

                    # Set bounds and initial guesses based on the peak type
                    bounds_L_low = np.array([0, small_range, 0])
                    bounds_L_high = np.array([np.inf, large_range, np.inf])
                    initial_guesses_L = fixed_initial_guesses[peak]["Lorentzian"]

                    # Fit the peak and plot
                    shifts_peak, fitted_curve, popt = fit_peak(
                        shifts, intensities[index], peak,
                        bounds_L_low, bounds_L_high, initial_guesses_L
                    )

                    if fitted_curve is not None:
                        plt.plot(shifts_peak, fitted_curve, '--', label=f'{peak} fit')

                        # Print details for debugging
                        print(f"Peak: {peak}")
                        print(f"A: {popt[0]}, P: {popt[1]}, W: {popt[2]}") # type: ignore
                        print(f"Max of fitted curve: {np.max(fitted_curve)}")
                        print(f"Max of raw data in this range: {np.max(intensities[index])}")

                # Adjust plot limits to make sure all data is visible
                plt.xlim(shifts.min(), shifts.max())
                plt.ylim(0, max_intensity)
                
                plt.title(f'Spectrum at x: {x_coordinates[index]:.2f}, y: {y_coordinates[index]:.2f} - Sample ID: {sample_id}')
                plt.xlabel('Raman Shift (cm-1)')
                plt.ylabel('Intensity')
                plt.legend()
                plt.show()

            plt.gcf().canvas.mpl_connect('button_press_event', on_click)
            plt.show()

            change_bounds = simpledialog.askstring("Axes", 'Would you like to change the axes? Input "Yes" or "No":')
            if change_bounds == 'Yes':
                            min_z = simpledialog.askfloat("Minimum z", "What is the smallest value of the z-axis?")
                            max_z = simpledialog.askfloat("Maximum z", "What is the largest value of the z-axis?")
            else:
                break

        # Calculate quartiles and IQR for outlier detection
    
        q1 = np.percentile(ratio_area, 25)
        q3 = np.percentile(ratio_area, 75)
        iqr = q3 - q1

            # Define bounds and filter outliers
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_data = [x for x in ratio_area if lower_bound <= x <= upper_bound]

            # Create histogram and calculate bin centers
        hist, bin_edges = np.histogram(filtered_data, bins=20)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Fit single Gaussian
        popt_single, pcov_single = curve_fit(gaussian, bin_centers, hist, 
                                                p0=[hist.max(), np.mean(filtered_data), np.std(filtered_data)])
        perr_single = np.sqrt(np.diag(pcov_single))

            # Fit double Gaussian
        initial_guess = [hist.max()/2, np.mean(filtered_data)-np.std(filtered_data), np.std(filtered_data)/2,
                            hist.max()/2, np.mean(filtered_data)+np.std(filtered_data), np.std(filtered_data)/2]
        popt_double, pcov_double = curve_fit(double_gaussian, bin_centers, hist, 
                                                p0=initial_guess, maxfev=100000)
        perr_double = np.sqrt(np.diag(pcov_double))

            # Calculate BIC for single and double Gaussian models
        n = len(hist)
        k_single = 3  # number of parameters for single Gaussian
        k_double = 6  # number of parameters for double Gaussian
        residuals_single = hist - gaussian(bin_centers, *popt_single)
        residuals_double = hist - double_gaussian(bin_centers, *popt_double)
        bic_single = n * np.log(np.sum(residuals_single**2) / n) + k_single * np.log(n)
        bic_double = n * np.log(np.sum(residuals_double**2) / n) + k_double * np.log(n)

            # Calculate ΔBIC and determine significance
        delta_bic = bic_double - bic_single
        if delta_bic < 2:
            significance = "Little to no evidence against the model with the lower BIC."
            probability = 1 / (1 + np.exp(delta_bic / 2))
        elif 2 <= delta_bic < 6:
            significance = "Positive evidence against the model with the higher BIC."
            probability = 1 / (1 + np.exp(delta_bic / 2))
        elif 6 <= delta_bic < 10:
            significance = "Strong evidence against the model with the higher BIC."
            probability = 1 / (1 + np.exp(delta_bic / 2))
        else:
            significance = "Very strong evidence against the model with the higher BIC."
            probability = 1 / (1 + np.exp(delta_bic / 2))

            # Plot results
        plt.figure(figsize=(12, 6))
        plt.hist(filtered_data, bins=20, edgecolor='black', alpha=0.7, label='Data')
        plt.plot(bin_centers, gaussian(bin_centers, *popt_single), 'r-', label='Single Gaussian Fit')
        plt.plot(bin_centers, double_gaussian(bin_centers, *popt_double), 'g-', label='Double Gaussian Fit')

            # Plot individual Gaussians from the double Gaussian fit
        gaussian1 = gaussian(bin_centers, popt_double[0], popt_double[1], popt_double[2])
        gaussian2 = gaussian(bin_centers, popt_double[3], popt_double[4], popt_double[5])
        plt.plot(bin_centers, gaussian1, 'b--', label='Gaussian 1')
        plt.plot(bin_centers, gaussian2, 'm--', label='Gaussian 2')

            # Set axis labels and title 
        plt.xlabel('A(2D)/A(G)')  
        plt.ylabel('Frequency')
        plt.title(f'A(2D)/A(G) Gaussian Fits Comparison - Sample ID: {sample_id}')
        plt.legend()
        plt.show()

            # Print fitting results and BIC comparison
        print(f"Single Gaussian BIC: {bic_single:.2f}")
        print(f"Double Gaussian BIC: {bic_double:.2f}")
        print(f"ΔBIC: {delta_bic:.2f} ({significance})")
        print(f"Probability that Single Gaussian is better: {probability:.2%}")

            # Print fitting results and include mean and SEM
        ratio_type = 'A(2D)/A(G)'
        mean_ratio = np.mean(filtered_data) #Filt data is AD/AG, takes the info from line540 (ratio_area)
        sem_ratio = np.std(filtered_data, ddof=1) / np.sqrt(len(filtered_data)) # ddof is degrees of freedom, value is 1 to calc sample stand deviation
        print(f"\nMean {ratio_type}: {mean_ratio:.2f} ± {sem_ratio:.2f}")

        print("Single Gaussian Fit Parameters:")
        print(f"Amplitude: {popt_single[0]:.2f} ± {perr_single[0]:.2f}")
        print(f"Centre: {popt_single[1]:.2f} ± {perr_single[1]:.2f}")
        print(f"Width: {popt_single[2]:.2f} ± {perr_single[2]:.2f}")
        print(f"BIC: {bic_single:.2f}")

        print("\nDouble Gaussian Fit Parameters:")
        print(f"Amplitude 1: {popt_double[0]:.2f} ± {perr_double[0]:.2f}")
        print(f"Centre 1: {popt_double[1]:.2f} ± {perr_double[1]:.2f}")
        print(f"Width 1: {popt_double[2]:.2f} ± {perr_double[2]:.2f}")
        print(f"Amplitude 2: {popt_double[3]:.2f} ± {perr_double[3]:.2f}")
        print(f"Centre 2: {popt_double[4]:.2f} ± {perr_double[4]:.2f}")
        print(f"Width 2: {popt_double[5]:.2f} ± {perr_double[5]:.2f}")
        print(f"BIC: {bic_double:.2f}")

root = tk.Tk()
root.title("Peak Analysis Tool")

frame = tk.Frame(root)
frame.pack(padx=20, pady=20)

btn = tk.Button(frame, text="Analyze Peaks", command=analyze_peaks)
btn.pack(padx=10, pady=10)

root.mainloop()
