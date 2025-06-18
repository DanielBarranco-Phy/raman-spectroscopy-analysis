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

    def fit_L(x, A, P, W): 
        term_1 = peakL(x, A, P, W)
        term_2 = backline(x)
        return term_1 + term_2

    def backline(x):
        backliney = (y_last - y_first) / (x_last - x_first) * (x - x_first) + y_first - const
        return backliney

    def normalize_data(data):  
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    
    def gaussian(x, amp, mu, sigma):
        return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

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
                    if try_again.lower() != 'yes': 
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
        root.withdraw() 

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
                
                A_err = np.sqrt(pcov[0][0]) 
            

                areas[peak].append(popt[0])
                peaks[peak].append(popt[1])

                if peak == 'D':
                    D_area = popt[0]
                    D_area_err = A_err
                elif peak == 'G':
                    G_area = popt[0]
                    G_area_err = A_err
                    
                   
                    if 'D' in areas:
                        ratio = D_area / G_area 
                        ratio_err = ratio * np.sqrt((D_area_err/D_area)**2 + (G_area_err/G_area)**2)
                        AD_AG_ratio.append(ratio) 
                        AD_AG_ratio_err.append(ratio_err) 


            except RuntimeError as e:
                print(f"An error occurred at index {i} for peak {peak}: {e}")
                dummy_area, dummy_peak = {
                    'G': (100, 100),
                    'D': (0.1, 100),
                    '2D': (0.1, 100),
                }.get(peak, (1e6, 1e6))  
                areas[peak].append(dummy_area)
                peaks[peak].append(dummy_peak)

            if i == peak_index:
                plt.figure()
                plt.plot(shifts_peak, intensities_peak, 'x')
                plt.plot(shifts_peak, fit_L(shifts_peak, *popt), color='black')
                plt.title(f'Sample ID: {sample_id}')
                plt.show()

    if 'G' in peaks:
        
        min_z = min(peaks['G']) 
        max_z = max(peaks['G'])
        while True:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            cmap = plt.get_cmap('viridis')
            sc = ax.scatter(x_coordinates, y_coordinates, peaks['G'], c=peaks['G'], cmap=cmap, marker='o', s=5000/len(x_coordinates), vmin=min_z, vmax=max_z, alpha=0.7) 
            cbar = fig.colorbar(sc)
            cbar.set_label('pos(G) / cm$^{-1}$')
            ax.set_xlabel('x-position / μm')
            ax.set_ylabel('y-position / μm')
            ax.set_zlabel('pos(G) / cm$^{-1}$') 
            ax.set_zlim3d(min_z, max_z) 
            ax.grid(visible=False)
            plt.title(f'pos(G) - Sample ID: {sample_id}')
            plt.show()

            change_bounds = simpledialog.askstring("Axes", 'Would you like to change the axes? Input "Yes" or "No":')
            if change_bounds == 'Yes':
                min_z = simpledialog.askfloat("Minimum z", "What is the smallest value of the z-axis?")
                max_z = simpledialog.askfloat("Maximum z", "What is the largest value of the z-axis?")
            else:
                break

        
        if len(x_coordinates) > 1 and len(y_coordinates) > 1:
            distance = np.sqrt((x_coordinates[1] - x_coordinates[0])**2 + (y_coordinates[1] - y_coordinates[0])**2)
            s = (distance * 0.8)**2  
        else:
            s = 20 

        # 2D plot of pos(G)
        xi, yi = np.meshgrid(np.linspace(min(x_coordinates), max(x_coordinates), 500),
                             np.linspace(min(y_coordinates), max(y_coordinates), 500))
        zi = griddata((x_coordinates.flatten(), y_coordinates.flatten()), np.array(peaks['G']), (xi, yi), method='nearest')

        
        
        while True:
            smoothed_zi = gaussian_filter(zi, sigma=2)

            
            contourf_levels = np.linspace(min_z, max_z, 400)  

            
            plt.contourf(xi, yi, smoothed_zi, contourf_levels, cmap='viridis')

            
            plt.colorbar(label='pos(G) / cm$^{-1}$')

            sc = plt.scatter(x_coordinates, y_coordinates, c=peaks['G'], s=s, alpha=0) 

            plt.gca().set_aspect('equal')
            plt.xlim([min(x_coordinates), max(x_coordinates)])
            plt.ylim([min(y_coordinates), max(y_coordinates)])
            
            plt.xlabel('x-position / μm')
            plt.ylabel('y-position / μm')
            plt.title(f'pos(G) - Sample ID: {sample_id}')
        

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

                
                y_first = np.mean(intensities_peak[-10:])
                y_last = np.mean(intensities_peak[:10])
                x_first = np.mean(shifts_peak[-10:])
                x_last = np.mean(shifts_peak[:10])
                const = (intensities_peak.max() - intensities_peak.min()) / 35

                
                initial_guesses_L = np.clip(initial_guesses_L, bounds_L_low, bounds_L_high)

                try:
                    popt, _ = optimize.curve_fit(
                        lambda x, A, P, W: fit_L(x, A, P, W, y_first, y_last, x_first, x_last, const),
                        shifts_peak, intensities_peak, p0=initial_guesses_L,
                        bounds=(bounds_L_low, bounds_L_high), maxfev=10000
                    )
                    fitted_curve = fit_L(shifts_peak, *popt, y_first, y_last, x_first, x_last, const)
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

                
                for peak in all_peaks:
                    small_range, large_range = fixed_ranges[peak]

                   
                    bounds_L_low = np.array([0, small_range, 0])
                    bounds_L_high = np.array([np.inf, large_range, np.inf])
                    initial_guesses_L = fixed_initial_guesses[peak]["Lorentzian"]

                   
                    shifts_peak, fitted_curve, popt = fit_peak(
                        shifts, intensities[index], peak,
                        bounds_L_low, bounds_L_high, initial_guesses_L
                    )

                    if fitted_curve is not None:
                        plt.plot(shifts_peak, fitted_curve, '--', label=f'{peak} fit')

                        
                        print(f"Peak: {peak}")
                        print(f"A: {popt[0]}, P: {popt[1]}, W: {popt[2]}")
                        print(f"Max of fitted curve: {np.max(fitted_curve)}")
                        print(f"Max of raw data in this range: {np.max(intensities[index])}")

              
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

         
            theo_x = np.linspace(1580, 1595, 1000)
            theo_y = theoliney(theo_x)

         
            plt.figure(figsize=(12, 6))

   
            plt.subplot(1, 2, 1)
            plt.scatter(peaks['G'], peaks['2D'], label='Data points')
            plt.plot(theo_x, theo_y, linestyle='--', color='red', label='Theoretical Line')
            plt.xlabel('pos(G) / cm$^{-1}$')
            plt.ylabel('pos(2D) / cm$^{-1}$')
            plt.title(f'pos(2D) against pos(G) - Sample ID: {sample_id}')
            plt.legend()

           
            plt.subplot(1, 2, 2)
            plt.scatter(peaks['G'], peaks['2D'], label='Data points')
            plt.plot(theo_x, theo_y, linestyle='--', color='red', label='Theoretical Line')
            plt.xlim(1575, 1605)
            plt.ylim(2720, 2800)
            plt.xlabel('pos(G) / cm$^{-1}$')
            plt.ylabel('pos(2D) / cm$^{-1}$')
            plt.title(f'pos(2D) against pos(G) - Sample ID: {sample_id} (Constant Scale)')
            plt.legend()

       
            plt.tight_layout()
            plt.show()

          
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
                ax.set_zlabel('pos(2D) / cm$^{-1}$') 
                ax.set_zlim3d(min_z, max_z) 
                ax.grid(visible=False)
                plt.title(f'pos(2D) - Sample ID: {sample_id}')
                plt.show()

                change_bounds = simpledialog.askstring("Axes", 'Would you like to change the axes? Input "Yes" or "No":')
                if change_bounds == 'Yes':
                    min_z = simpledialog.askfloat("Minimum z","What is the smallest value of the z-axis?")
                    max_z = simpledialog.askfloat("Maximum z","What is the largest value of the z-axis?")
                else:
                    break
         
            distance = np.sqrt((x_coordinates[1] - x_coordinates[0])**2 + (y_coordinates[1] - y_coordinates[0])**2)

           
            s = (distance * 0.8)**2
        
            # 2D plot of pos(2D)
            xi, yi = np.meshgrid(np.linspace(min(x_coordinates), max(x_coordinates), 500),
                                np.linspace(min(y_coordinates), max(y_coordinates), 500))
            zi = griddata((x_coordinates.flatten(), y_coordinates.flatten()), np.array(peaks['2D']), (xi, yi), method='nearest')

            while True:
                smoothed_zi = gaussian_filter(zi, sigma=2)
                levels = np.linspace(min_z, max_z, 400)

                plt.contourf(xi, yi, smoothed_zi, levels, cmap='plasma')
                plt.colorbar(label='pos(2D) / cm$^{-1}$')
                plt.scatter(x_coordinates, y_coordinates, c=peaks['2D'], cmap='Greys', marker='s', s=0, vmin=min_z, vmax=max_z, edgecolors='none')
                plt.gca().set_aspect('equal')
             
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

         
                    y_first = np.mean(intensities_peak[-10:])
                    y_last = np.mean(intensities_peak[:10])
                    x_first = np.mean(shifts_peak[-10:])
                    x_last = np.mean(shifts_peak[:10])
                    const = (intensities_peak.max() - intensities_peak.min()) / 35

   
                    initial_guesses_L = np.clip(initial_guesses_L, bounds_L_low, bounds_L_high)

                    try:
                        popt, _ = optimize.curve_fit(
                            lambda x, A, P, W: fit_L(x, A, P, W, y_first, y_last, x_first, x_last, const),
                            shifts_peak, intensities_peak, p0=initial_guesses_L,
                            bounds=(bounds_L_low, bounds_L_high), maxfev=10000
                        )
                        fitted_curve = fit_L(shifts_peak, *popt, y_first, y_last, x_first, x_last, const)
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

       
                    for peak in all_peaks:
                        small_range, large_range = fixed_ranges[peak]

                     
                        bounds_L_low = np.array([0, small_range, 0])
                        bounds_L_high = np.array([np.inf, large_range, np.inf])
                        initial_guesses_L = fixed_initial_guesses[peak]["Lorentzian"]

                       
                        shifts_peak, fitted_curve, popt = fit_peak(
                            shifts, intensities[index], peak,
                            bounds_L_low, bounds_L_high, initial_guesses_L
                        )

                        if fitted_curve is not None:
                            plt.plot(shifts_peak, fitted_curve, '--', label=f'{peak} fit')

                
                            print(f"Peak: {peak}")
                            print(f"A: {popt[0]}, P: {popt[1]}, W: {popt[2]}") 
                            print(f"Max of fitted curve: {np.max(fitted_curve)}")
                            print(f"Max of raw data in this range: {np.max(intensities[index])}") 

                 
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
                sc = ax.scatter(x_coordinates, y_coordinates, ratio_area, c=ratio_area, cmap='viridis', marker='o', s=5000/len(x_coordinates), vmin=min_z, vmax=max_z, alpha=0.7) 
                cbar = fig.colorbar(sc)
                cbar.set_label('A(D)/A(G)')
                ax.set_xlabel('x-position / μm')
                ax.set_ylabel('y-position / μm')
                ax.set_zlabel('A(D)/A(G)') 
                ax.set_zlim3d(min_z, max_z) 
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
  
                smoothed_zi = gaussian_filter(zi, sigma=2)

                levels = np.linspace(min_z, max_z, 400)

                plt.contourf(xi, yi, smoothed_zi, levels, cmap='viridis')
                plt.colorbar(label='A(D)/A(G)')
                plt.scatter(x_coordinates, y_coordinates, c=ratio_area, cmap='Greys', marker='s', s=0, vmin=min_z, vmax=max_z, edgecolors='none')
                plt.gca().set_aspect('equal')
 
                plt.xlim([min(x_coordinates), max(x_coordinates)])
                plt.ylim([min(y_coordinates), max(y_coordinates)])
                plt.xlabel('x-position / μm')
                plt.ylabel('y-position / μm')
                plt.title(f'A(D)/A(G) - Sample ID: {sample_id}')

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

                    
                    y_first = np.mean(intensities_peak[-10:])
                    y_last = np.mean(intensities_peak[:10])
                    x_first = np.mean(shifts_peak[-10:])
                    x_last = np.mean(shifts_peak[:10])
                    const = (intensities_peak.max() - intensities_peak.min()) / 35


                    initial_guesses_L = np.clip(initial_guesses_L, bounds_L_low, bounds_L_high)

                    try:
                        popt, _ = optimize.curve_fit(
                            lambda x, A, P, W: fit_L(x, A, P, W, y_first, y_last, x_first, x_last, const),
                            shifts_peak, intensities_peak, p0=initial_guesses_L,
                            bounds=(bounds_L_low, bounds_L_high), maxfev=10000
                        )
                        fitted_curve = fit_L(shifts_peak, *popt, y_first, y_last, x_first, x_last, const)
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

                  
                    for peak in all_peaks:
                        small_range, large_range = fixed_ranges[peak]

                      
                        bounds_L_low = np.array([0, small_range, 0])
                        bounds_L_high = np.array([np.inf, large_range, np.inf])
                        initial_guesses_L = fixed_initial_guesses[peak]["Lorentzian"]

                  
                        shifts_peak, fitted_curve, popt = fit_peak(
                            shifts, intensities[index], peak,
                            bounds_L_low, bounds_L_high, initial_guesses_L
                        )

                        if fitted_curve is not None:
                            plt.plot(shifts_peak, fitted_curve, '--', label=f'{peak} fit')

               
                            print(f"Peak: {peak}")
                            print(f"A: {popt[0]}, P: {popt[1]}, W: {popt[2]}")
                            print(f"Max of fitted curve: {np.max(fitted_curve)}")
                            print(f"Max of raw data in this range: {np.max(intensities[index])}")

   
                    plt.xlim(shifts.min(), shifts.max())
                    plt.ylim(0, max_intensity)
                    plt.title(f'Spectrum at x: {x_coordinates[index]:.2f}, y: {y_coordinates[index]:.2f} - Sample ID: {sample_id}')
                    plt.xlabel('Raman Shift (cm-1)')
                    plt.ylabel('Intensity')
                    plt.legend()
                    plt.show() 

                    if 'D' in peaks and 'G' in peaks:
                        ratio = AD_AG_ratio[index]
                        ratio_err = AD_AG_ratio_err[index]
                        messagebox.showinfo("A(D)/A(G) Ratio", f"A(D)/A(G) ratio at this point: {ratio:.2f} ± {ratio_err:.2f}")   
                        


                plt.gcf().canvas.mpl_connect('button_press_event', on_click)
                plt.show()

                

                change_bounds = simpledialog.askstring("Axes", 'Would you like to change the axes? Input "Yes" or "No":')
                if change_bounds == 'Yes':
                        min_z = simpledialog.askfloat("Minimum z", "What is the smallest value of the z-axis?")
                        max_z = simpledialog.askfloat("Maximum z", "What is the largest value of the z-axis?")
                else:
                    break


           
            q1 = np.percentile(ratio_area, 25)
            q3 = np.percentile(ratio_area, 75)
            iqr = q3 - q1

       
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            filtered_data = [x for x in ratio_area if lower_bound <= x <= upper_bound]

     
            hist, bin_edges = np.histogram(filtered_data, bins=20)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            
            popt_single, pcov_single = curve_fit(gaussian, bin_centers, hist, 
                                                p0=[hist.max(), np.mean(filtered_data), np.std(filtered_data)])
            perr_single = np.sqrt(np.diag(pcov_single))

           
            initial_guess = [hist.max()/2, np.mean(filtered_data)-np.std(filtered_data), np.std(filtered_data)/2,
                            hist.max()/2, np.mean(filtered_data)+np.std(filtered_data), np.std(filtered_data)/2]
            popt_double, pcov_double = curve_fit(double_gaussian, bin_centers, hist, 
                                                p0=initial_guess, maxfev=100000)
            perr_double = np.sqrt(np.diag(pcov_double))

            
            n = len(hist)
            k_single = 3  
            k_double = 6 
            residuals_single = hist - gaussian(bin_centers, *popt_single)
            residuals_double = hist - double_gaussian(bin_centers, *popt_double)
            bic_single = n * np.log(np.sum(residuals_single**2) / n) + k_single * np.log(n)
            bic_double = n * np.log(np.sum(residuals_double**2) / n) + k_double * np.log(n)

 
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

          
            plt.figure(figsize=(12, 6))
            plt.hist(filtered_data, bins=20, edgecolor='black', alpha=0.7, label='Data')
            plt.plot(bin_centers, gaussian(bin_centers, *popt_single), 'r-', label='Single Gaussian Fit')
            plt.plot(bin_centers, double_gaussian(bin_centers, *popt_double), 'g-', label='Double Gaussian Fit')


            gaussian1 = gaussian(bin_centers, popt_double[0], popt_double[1], popt_double[2])
            gaussian2 = gaussian(bin_centers, popt_double[3], popt_double[4], popt_double[5])
            plt.plot(bin_centers, gaussian1, 'b--', label='Gaussian 1')
            plt.plot(bin_centers, gaussian2, 'm--', label='Gaussian 2')

       
            plt.xlabel('A(D)/A(G)')
            plt.ylabel('Frequency')
            plt.title(f'A(D)/A(G) Gaussian Fits Comparison - Sample ID: {sample_id}')
            plt.legend()
            plt.show()

        
            print(f"Single Gaussian BIC: {bic_single:.2f}")
            print(f"Double Gaussian BIC: {bic_double:.2f}")
            print(f"ΔBIC: {delta_bic:.2f} ({significance})")
            print(f"Probability that Single Gaussian is better: {probability:.2%}")


            ratio_type = 'A(D)/A(G)'
            mean_ratio = np.mean(filtered_data) 
            sem_ratio = np.std(filtered_data, ddof=1) / np.sqrt(len(filtered_data)) 
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


    if 'G' in peaks and '2D' in peaks:
        ratio_area = np.array(areas['2D'])/np.array(areas['G'])
        min_z = min(ratio_area)
        max_z = max(ratio_area)  
        while True:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            cmap = plt.colormaps.get_cmap('Greys')
            sc = ax.scatter(x_coordinates, y_coordinates, ratio_area, c=ratio_area, cmap='plasma', marker='o', s=5000/len(x_coordinates), vmin=min_z, vmax=max_z, alpha=0.7) 
            cbar = fig.colorbar(sc)
            cbar.set_label('A(2D)/A(G)')
            ax.set_xlabel('x-position / μm')
            ax.set_ylabel('y-position / μm')
            ax.set_zlabel('A(2D)/A(G)')
            ax.set_zlim3d(min_z, max_z)
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
        
            smoothed_zi = gaussian_filter(zi, sigma=2)

            levels = np.linspace(min_z, max_z, 400) 

            plt.contourf(xi, yi, smoothed_zi, levels, cmap='plasma')
            plt.colorbar(label='A(2D)/A(G)')
            plt.scatter(x_coordinates, y_coordinates, c=ratio_area, cmap='Greys', marker='s', s=0, vmin=min_z, vmax=max_z, edgecolors='none')
            plt.gca().set_aspect('equal')

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


                y_first = np.mean(intensities_peak[-10:])
                y_last = np.mean(intensities_peak[:10])
                x_first = np.mean(shifts_peak[-10:])
                x_last = np.mean(shifts_peak[:10])
                const = (intensities_peak.max() - intensities_peak.min()) / 35

                
                initial_guesses_L = np.clip(initial_guesses_L, bounds_L_low, bounds_L_high)

                try:
                    popt, _ = optimize.curve_fit(
                        lambda x, A, P, W: fit_L(x, A, P, W, y_first, y_last, x_first, x_last, const),
                        shifts_peak, intensities_peak, p0=initial_guesses_L,
                        bounds=(bounds_L_low, bounds_L_high), maxfev=10000
                    )
                    fitted_curve = fit_L(shifts_peak, *popt, y_first, y_last, x_first, x_last, const)
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

    
                for peak in all_peaks:
                    small_range, large_range = fixed_ranges[peak]

                    bounds_L_low = np.array([0, small_range, 0])
                    bounds_L_high = np.array([np.inf, large_range, np.inf])
                    initial_guesses_L = fixed_initial_guesses[peak]["Lorentzian"]

             
                    shifts_peak, fitted_curve, popt = fit_peak(
                        shifts, intensities[index], peak,
                        bounds_L_low, bounds_L_high, initial_guesses_L
                    )

                    if fitted_curve is not None:
                        plt.plot(shifts_peak, fitted_curve, '--', label=f'{peak} fit')

                        print(f"Peak: {peak}")
                        print(f"A: {popt[0]}, P: {popt[1]}, W: {popt[2]}")
                        print(f"Max of fitted curve: {np.max(fitted_curve)}")
                        print(f"Max of raw data in this range: {np.max(intensities[index])}")

        
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

       
    
        q1 = np.percentile(ratio_area, 25)
        q3 = np.percentile(ratio_area, 75)
        iqr = q3 - q1

            
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_data = [x for x in ratio_area if lower_bound <= x <= upper_bound]

      
        hist, bin_edges = np.histogram(filtered_data, bins=20)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

           
        popt_single, pcov_single = curve_fit(gaussian, bin_centers, hist, 
                                                p0=[hist.max(), np.mean(filtered_data), np.std(filtered_data)])
        perr_single = np.sqrt(np.diag(pcov_single))

          
        initial_guess = [hist.max()/2, np.mean(filtered_data)-np.std(filtered_data), np.std(filtered_data)/2,
                            hist.max()/2, np.mean(filtered_data)+np.std(filtered_data), np.std(filtered_data)/2]
        popt_double, pcov_double = curve_fit(double_gaussian, bin_centers, hist, 
                                                p0=initial_guess, maxfev=100000)
        perr_double = np.sqrt(np.diag(pcov_double))

  
        n = len(hist)
        k_single = 3 
        k_double = 6 
        residuals_single = hist - gaussian(bin_centers, *popt_single)
        residuals_double = hist - double_gaussian(bin_centers, *popt_double)
        bic_single = n * np.log(np.sum(residuals_single**2) / n) + k_single * np.log(n)
        bic_double = n * np.log(np.sum(residuals_double**2) / n) + k_double * np.log(n)

            
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

           
        plt.figure(figsize=(12, 6))
        plt.hist(filtered_data, bins=20, edgecolor='black', alpha=0.7, label='Data')
        plt.plot(bin_centers, gaussian(bin_centers, *popt_single), 'r-', label='Single Gaussian Fit')
        plt.plot(bin_centers, double_gaussian(bin_centers, *popt_double), 'g-', label='Double Gaussian Fit')

        
        gaussian1 = gaussian(bin_centers, popt_double[0], popt_double[1], popt_double[2])
        gaussian2 = gaussian(bin_centers, popt_double[3], popt_double[4], popt_double[5])
        plt.plot(bin_centers, gaussian1, 'b--', label='Gaussian 1')
        plt.plot(bin_centers, gaussian2, 'm--', label='Gaussian 2')

         
        plt.xlabel('A(2D)/A(G)')  
        plt.ylabel('Frequency')
        plt.title(f'A(2D)/A(G) Gaussian Fits Comparison - Sample ID: {sample_id}')
        plt.legend()
        plt.show()

        print(f"Single Gaussian BIC: {bic_single:.2f}")
        print(f"Double Gaussian BIC: {bic_double:.2f}")
        print(f"ΔBIC: {delta_bic:.2f} ({significance})")
        print(f"Probability that Single Gaussian is better: {probability:.2%}")

  
        ratio_type = 'A(2D)/A(G)'
        mean_ratio = np.mean(filtered_data) 
        sem_ratio = np.std(filtered_data, ddof=1) / np.sqrt(len(filtered_data)) 
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
