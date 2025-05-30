import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

def read_data():
    filepaths = filedialog.askopenfilenames(title="Select CSV files", filetypes=[("CSV files", "*.csv")])
    if not filepaths:
        return None
    dfs = [pd.read_csv(filepath, header=None) for filepath in filepaths]
    return dfs

def convert_np(an_array):
    values = [float(x) for x in an_array.split()] # converts data in array so program can analyze it 
    return np.array(values)

def find_fwhm(x, y):
    
    half_max = np.max(y) / 2 # This calculates half of the maximum y-value in the dataset. This is the "Half Maximum" part of FWHM.
    left_idx = np.argmax(y >= half_max) # This finds the index of the first point where the y-value is greater than or equal to half_max.
    # y >= half_max creates a boolean array where True indicates y-values above half_max.
    #returns the index of the first True value.

    right_idx = len(y) - np.argmax(y[::-1] >= half_max) - 1 # This finds the index of the last point where the y-value is greater than or equal to half_max.
    # y[::-1] reverses the y array.
    # np.argmax(y[::-1] >= half_max) finds the first index from the end where y is above half_max.
    # len  (y) - ... - 1 converts this to the correct index in the original array order.

    return x[right_idx] - x[left_idx] # This calculates the difference between the x-values at the right and left indices.
    #This difference is the width of the peak at half its maximum height, which is the FWHM.


def calculate_fwhm(fit_type, params):
        if fit_type == "Lorentzian":
            return 2 * params[2] # represents the width parameter 
        elif fit_type == "Gaussian":
            return 2 * np.sqrt(2 * np.log(2)) * params[2] # represents σ. mathematical relationship between σ and FWHM for a Gaussian function.
        elif fit_type == "Voigt":
            lorentzian_width = params[2]
            gaussian_width = params[3]
            #is the Lorentzian width, and params[3] is the Gaussian width.
            return 0.5346 * lorentzian_width + np.sqrt(0.2166 * lorentzian_width**2 + gaussian_width**2)
        elif fit_type == "2 Lorentzians":
            if params[0] > params[3]:
                return 2 * params[2]
            else:
                return 2 * params[5]
            #params[0] and params[3] are the amplitudes of the two peaks. 
            # If the first peak is stronger, we use its width (params[2]), 
            # otherwise we use the width of the second peak (params[5]).
        else:
            return None

def calculate_dynamic_initial_guess_area(shifts, intensities, peak_range):  # calculates an initial guess for the peak area based on the data within a specified range.
    peak_indices = np.where((shifts >= peak_range[0]) & (shifts <= peak_range[1])) # filters data that falls between peak range, de donde a donde van los picos 
    area = np.trapezoid(intensities[peak_indices], shifts[peak_indices])#Calculates the area under the best-fit curve using built in function trapezoid, area between data 
    return area # np.trapz approximates the integral of y-values (intensities) with respect to x-values (shifts) using the trapezoidal rule.


def analyze_peaks():
    dfs = read_data()
    if dfs is None:
        return

    peak_data = {'D': [], 'G': [], '2D': []}
    intensity_data = {'D': [], 'G': [], '2D': []}
    fwhm_data = {'D': [], 'G': [], '2D': []}

    unique_peaks = set()
    while True:
        peak = simpledialog.askstring("Peaks", 'Name a peak that you want to analyse. If you have inputted all the peaks, input "No":')
        if peak in {'D', 'G', '2D'}:
            unique_peaks.add(peak)
        elif peak == "No" or peak is None:
            break

    all_peaks = list(unique_peaks)

    def peakL(x, A1, P1, W1):
        return A1 * W1 / (np.pi * ((x - P1)**2 + W1**2))

    def peakG(x, A1, P1, S1):
        return A1 * np.exp(-((x - P1)**2) / (2 * S1**2))

    def peakV(x, A1, P1, W1, S1):
        return (1 - S1) * peakL(x, A1, P1, W1) + S1 * peakG(x, A1, P1, S1)

    def fit_L(x, A, P, W):
        return peakL(x, A, P, W) + backline(x)

    def fit_G(x, A, P, S):
        return peakG(x, A, P, S) + backline(x)

    def fit_V(x, A, P, W, S):
        return peakV(x, A, P, W, S) + backline(x)

    def fit_LL(x, A1, P1, W1, A2, P2, W2):
        return peakL(x, A1, P1, W1) + peakL(x, A2, P2, W2) + backline(x)

    def backline(x): #background line function is also defined to account for baseline shifts. 
        return (y_last - y_first) / (x_last - x_first) * (x - x_first) + y_first - const

    for i, df in enumerate(dfs):
        shifts = np.array(df.iloc[:, 0].values) # extracts the first column of the DataFrame. They are in this order because when converted data into array, data is organized in this way
        intensities = np.array(df.iloc[:, 1].values) # extracts the second column of the DataFrame.

        fixed_ranges = {
            'D': (1250, 1400),
            'G': (1450, 1700),
            '2D': (2600, 2800)
        }
        #Loops through each DataFrame.
        #Extracts shifts (the x-axis values) and intensities (the y-axis values) from the DataFrame.
        #Defines fixed ranges for different peaks to narrow down the regions of interest in the data.
            


        #Peak Fitting Functions
        #Define Various Peak Functions and Fit Functions:
        # creates imaginary functions to simulate desired funtion to later check which one is the best
        
    for peak in all_peaks:
        try:
                
            #retrieves the predefined range for each peak type. If the peak isn't recognized, it defaults to (0.0, 0.0) and shows a warning.
                small_range, large_range = fixed_ranges.get(peak, (0.0, 0.0))
                if small_range == 0.0 and large_range == 0.0:
                    messagebox.showwarning("Invalid Peak Type", f"The peak type '{peak}' is not recognized.")
                    continue
            #np.ma.masked_array: Masks values outside the desired range to find the smallest and largest indices within the peak range. Solo muestra valores dentro del peak range 
                mask_data_x_s = np.ma.masked_array(shifts, shifts <= small_range) # Reject things smaller than small range
                smallest = np.ma.argmin(mask_data_x_s)
            # A masked array is an array where certain values are treated as invalid or hidden.
                mask_data_x_l = np.ma.masked_array(shifts, shifts >= large_range) # Reject things higher than large range
                largest = np.ma.argmax(mask_data_x_l)
            #Extracts shifts_peak and intensities_peak within the identified range.
                shifts_peak = shifts[smallest:largest + 1] 
                intensities_peak = intensities[smallest:largest + 1]
            #Computes the linear background using the average of the first and last 5 points.
                y_first = np.mean(intensities_peak[-5:])
                y_last = np.mean(intensities_peak[:5])
                x_first = np.mean(shifts_peak[-5:])
                x_last = np.mean(shifts_peak[:5])
                
                const = (intensities_peak.max() - intensities_peak.min()) / 35 
            #const is a small constant added to the linear background to stabilize the fitting process.
            # boundaries of all fit types
                        
               
                const = (intensities_peak.max() - intensities_peak.min()) / 35 
                bounds_L_low = [0, small_range, 0]
                bounds_L_high = [np.inf, large_range, np.inf]
                bounds_G_low = [0, small_range, 0]
                bounds_G_high = [np.inf, large_range, np.inf]
                bounds_V_low = [0, small_range, 0, 0]
                bounds_V_high = [np.inf, large_range, np.inf, np.inf]
                bounds_LL_low = [0, small_range, 0, 0, small_range, 0]
                bounds_LL_high = [np.inf, large_range, np.inf, np.inf, large_range, np.inf]
                
                peak_range = fixed_ranges.get(peak, (0.0, 0.0))
                if peak_range == (0.0, 0.0):
                    messagebox.showwarning("Invalid Peak Type", f"The peak type '{peak}' is not recognized.")
                    continue

                G_initial_area = calculate_dynamic_initial_guess_area(shifts_peak, intensities_peak, peak_range)
                TwoD_initial_area = calculate_dynamic_initial_guess_area(shifts_peak, intensities_peak, peak_range)

                fixed_initial_guesses = {
                    'D': {
                        "Lorentzian": [720, 1350, 20],
                        "Gaussian": [720, 1350, 20],
                        "Voigt": [720, 1350, 20, 20],
                        "2 Lorentzians": [720, 1350, 20, 720, 1340, 20]
                    },
                    'G': {
                        "Lorentzian": [G_initial_area, 1580, 15],
                        "Gaussian": [G_initial_area, 1580, 15],
                        "Voigt": [G_initial_area, 1580, 15, 15],
                        "2 Lorentzians": [G_initial_area, 1580, 15, 500, 1600, 15]
                    },
                    '2D': {
                        "Lorentzian": [TwoD_initial_area, 2700, 30],
                        "Gaussian": [TwoD_initial_area, 2700, 30],
                        "Voigt": [TwoD_initial_area, 2700, 30, 30],
                        "2 Lorentzians": [TwoD_initial_area, 2700, 30, 700, 2680, 30]
                    }
                }

                initial_guess = fixed_initial_guesses.get(peak, {}).get("Lorentzian", [])

                fitting_functions = [
                    (fit_L, bounds_L_low, bounds_L_high, initial_guess, "Lorentzian"),
                    (fit_G, bounds_G_low, bounds_G_high, fixed_initial_guesses[peak]["Gaussian"], "Gaussian"),
                    (fit_V, bounds_V_low, bounds_V_high, fixed_initial_guesses[peak]["Voigt"], "Voigt"),
                    (fit_LL, bounds_LL_low, bounds_LL_high, fixed_initial_guesses[peak]["2 Lorentzians"], "2 Lorentzians")
                ]
                #initial_guess: Retrieves the initial guess for the Lorentzian fit.
                #fitting_functions: List of tuples, each containing a fitting function, 
                # its bounds, initial guesses, and a label. 
                # This list is used to try different fitting models on the peak data.

                
                best_fit = None
                best_params = None
                best_fit_type = None

                bics = []

                for fit_func, bounds_low, bounds_high, initial_guess, fit_type in fitting_functions:
                    try:
                        popt, _ = optimize.curve_fit(fit_func, shifts_peak, intensities_peak, p0=initial_guess, bounds=(bounds_low, bounds_high))
                        #Uses scipy's curve_fit to find the best parameters for the current fitting function.
                        #popt contains the optimized parameters.
                        #p0=initial provides initial guesses for the parameters.
                        #bounds sets the lower and upper limits for each parameter.
                        residuals = intensities_peak - fit_func(shifts_peak, *popt)
                        ss_res = np.sum(residuals**2) # Calculates the difference between the actual data and the fit, then sums the squares of these differences.
                        ss_tot = np.sum((intensities_peak - np.mean(intensities_peak))**2)
                        r2 = 1 - (ss_res / ss_tot)
                        # Computes the BIC, which balances goodness of fit with model complexity.
                        #Lower BIC indicates a better model.
                        bic = len(shifts_peak) * np.log(ss_res / len(shifts_peak)) + len(popt) * np.log(len(shifts_peak))
                        bics.append((bic, fit_func, popt, fit_type))

                    except Exception as e:
                        print(f"Fit for {fit_type} failed: {e}")

                # Sort BIC values in ascending order
                bics.sort(key=lambda x: x[0])  # Sort by the BIC values

                # Print BIC values along with their corresponding labels
                print(f"Sample {i + 1}, Peak {peak}:")
                for bic_value, _, _, label in bics:
                    print(f"{label}: BIC = {bic_value}")

                # Extract the top 3 BIC values
                top_3_bics = bics[:3]
                print(f"Top 3 BIC values: {[b[3] for b in top_3_bics]}")

                # Update the best fit if necessary
                if bics:
                    lowest_bic, best_fit, best_params, best_fit_type = min(bics, key=lambda x: x[0])

                    peak_area = np.trapezoid(best_fit(shifts_peak, *best_params) - backline(shifts_peak), shifts_peak) #Calculates the area under the best-fit curve using 
                    #built in function trapezoid, subtracting the background.
                    peak_data[peak].append(peak_area)

                    peak_intensity = np.max(best_fit(shifts_peak, *best_params) - backline(shifts_peak)) # The function best_fit models the expected intensity values over the range shifts_peak using the parameters provided in best_params. 
                    #Iex:
                    # if best_fit is a Gaussian function, best_params might include the amplitude, center, and width of the peak. The result 
                    # is an array of fitted intensity values corresponding to each x-value in shifts_peak.
                    # - backline to take into consideration start point of the data
                    intensity_data[peak].append(peak_intensity)

                    fwhm_value = calculate_fwhm(best_fit_type, best_params)

                    if fwhm_value is None:

                        fitted_y = best_fit(shifts_peak, *best_params) - backline(shifts_peak)
                        fwhm_value = find_fwhm(shifts_peak, fitted_y)
                        #fitted_y: Calculate the fitted intensity values minus the background (using best_fit and backline functions).
                        #find_fwhm(shifts_peak, fitted_y): Determine the FWHM based on the intensity values and the shifts.

                    fwhm_data[peak].append(fwhm_value)

                    plt.figure()
                    plt.plot(shifts_peak, intensities_peak, 'b-', label='Data')
                    plt.plot(shifts_peak, best_fit(shifts_peak, *best_params), 'r--', label='Best fit')
                    plt.xlabel('Shift')
                    plt.ylabel('Intensity')
                    plt.title(f'Sample {i + 1} - {peak} peak')
                    plt.legend()
                    plt.show()

        except Exception as e:
            print(f"Error processing peak {peak} in sample {i + 1}: {e}")

    for peak in peak_data.keys():
        if len(peak_data[peak]) > 0:
            average_area = np.mean(peak_data[peak])
            average_intensity = np.mean(intensity_data[peak])
            average_fwhm = np.mean(fwhm_data[peak]) if peak in fwhm_data and len(fwhm_data[peak]) > 0 else None
            
            fwhm_message = f"Average FWHM: {average_fwhm:.2f}" if average_fwhm is not None else ""
            messagebox.showinfo(f"{peak} Peak Info", 
                                f"Average area of {peak} peak: {average_area:.2f}\n"
                                f"Average intensity of {peak} peak: {average_intensity:.2f}\n"
                                f"{fwhm_message}")

    if 'D' in peak_data and 'G' in peak_data and len(peak_data['D']) > 0 and len(peak_data['G']) > 0:
        D_G_area_ratio = np.mean(peak_data['D']) / np.mean(peak_data['G'])
        D_G_intensity_ratio = np.mean(intensity_data['D']) / np.mean(intensity_data['G'])
        messagebox.showinfo("D/G Ratios", 
                            f"Average D/G area ratio: {D_G_area_ratio:.2f}\n"
                            f"Average I(D)/I(G) intensity ratio: {D_G_intensity_ratio:.2f}")

    if '2D' in peak_data and 'G' in peak_data and len(peak_data['2D']) > 0 and len(peak_data['G']) > 0:
        TwoD_G_area_ratio = np.mean(peak_data['2D']) / np.mean(peak_data['G'])
        TwoD_G_intensity_ratio = np.mean(intensity_data['2D']) / np.mean(intensity_data['G'])
        average_2D_fwhm = np.mean(fwhm_data['2D']) if '2D' in fwhm_data and len(fwhm_data['2D']) > 0 else None
        
        fwhm_message = f"Average 2D FWHM: {average_2D_fwhm:.2f}" if average_2D_fwhm is not None else ""
        messagebox.showinfo("2D/G Ratios and 2D FWHM", 
                            f"Average 2D/G area ratio: {TwoD_G_area_ratio:.2f}\n"
                            f"Average I(2D)/I(G) intensity ratio: {TwoD_G_intensity_ratio:.2f}\n"
                            f"{fwhm_message}")

root = tk.Tk()
root.title("Single Spectra Analysis Tool")

frame = tk.Frame(root)
frame.pack(padx=20, pady=20)

btn = tk.Button(frame, text="Analyze Peaks", command=analyze_peaks)
btn.pack(padx=10, pady=10)

root.mainloop()
