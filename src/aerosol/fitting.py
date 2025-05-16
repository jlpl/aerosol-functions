from sklearn.mixture import GaussianMixture
from kneed import KneeLocator
import numpy as np
import pandas as pd
import aerosol.functions as af
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
import time
from sklearn.metrics import mean_squared_error
from scipy.signal import savgol_filter

def to_meters(x):
    return (10**x)*1e-9

def fit_gmm_and_aic(n, samples):
    gmm = GaussianMixture(n_components=n)
    gmm.fit(samples.reshape(-1,1))
    return gmm.aic(samples.reshape(-1,1))

def gaussian(x, amplitude, mean, sigma):
    return amplitude * 1.0/(sigma * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * ((x - mean) / sigma) ** 2.0)

def fit_gmm(samples,n_components,coef):

    gmm = GaussianMixture(n_components=n_components)
    
    gmm.fit(samples.reshape(-1,1))

    weights = gmm.weights_
    means = gmm.means_[:,0]
    stddevs = np.array([np.sqrt(c[0,0]) for c in gmm.covariances_])
    
    gaussians = []
    for i in range(n_components):
        
        gaussian = {
            "mean":means[i],
            "sigma":stddevs[i],
            "amplitude":weights[i]*coef,
        }
        gaussians.append(gaussian)

    return gaussians

def multimodal_gaussian(x, *params):
    n_components = len(params) // 3
    y = np.zeros_like(x)
    for i in range(n_components):
        amplitude = params[i * 3]
        mean = params[i * 3 + 1]
        sigma = params[i * 3 + 2]
        y += gaussian(x, amplitude, mean, sigma)
    return y

def calc_pred(x,gaussians):
    pred = np.zeros(len(x))
    for g in gaussians:
        pred = pred + gaussian(x, g["amplitude"], g["mean"], g["sigma"])

    return list(pred)

def fit_multimodal_gaussian(data_x, data_y, gaussians):
    initial_guesses = []
    lower_bounds = []
    upper_bounds = []

    for g in gaussians:

        initial_guesses.append(g["amplitude"])
        initial_guesses.append(g["mean"]) 
        initial_guesses.append(g["sigma"])

        lower_bounds.append(1)
        lower_bounds.append(-np.inf)
        lower_bounds.append(-np.inf)        
        upper_bounds.append(np.inf)
        upper_bounds.append(np.inf)
        upper_bounds.append(np.inf)

    n_components = len(initial_guesses) // 3

    try:
        params, _ = curve_fit(
            multimodal_gaussian,
            data_x,
            data_y,
            p0=initial_guesses,
            bounds=(lower_bounds,upper_bounds),
        )
    except:
        return None

    gaussians = []
    for i in range(n_components):
        amplitude = params[i * 3]
        mean = params[i * 3 + 1]
        sigma = params[i * 3 + 2]
        gaussian = {
            "mean":mean,
            "sigma":sigma,
            "amplitude":amplitude
        }
        gaussians.append(gaussian)

    return gaussians

def mse(x,x_pred):
    return np.sum((x - x_pred)**2)

def calc_pred_gaussians(x,gaussians):
    pred_gaussians = []
    for g in gaussians:
        pred_gaussian = list(gaussian(x,g["amplitude"],g["mean"],g["sigma"]))
        pred_gaussians.append(pred_gaussian)
    return pred_gaussians
    
def get_peak_positions(gaussians):
    dp = []
    for g in gaussians:
        dp.append(to_meters(g["mean"]))
    return dp

def calc_conc_ndist(x,ndist):
    conc = af.calc_conc(
            pd.Series(index = to_meters(x),data=ndist).to_frame().transpose(),
            to_meters(x.min()),
            to_meters(x.max())).iloc[0,0]
    return conc

def calc_conc_gaus(x,gaussians):
    mode_concs = []
    for g in gaussians:
        pred_gaussian = list(gaussian(x,g["amplitude"],g["mean"],g["sigma"]))
        pred_gaussian = pd.Series(index=to_meters(x), data = pred_gaussian)
        xmin = g["mean"] - 5 * g["sigma"]
        xmax = g["mean"] + 5 * g["sigma"]
        mode_conc = af.calc_conc(
                pd.Series(index = to_meters(x),data=pred_gaussian).to_frame().transpose(),
                to_meters(xmin),
                to_meters(xmax)).iloc[0,0]
        mode_concs.append(mode_conc)
    return mode_concs

def find_final_gaussians(gaussians_gmm, gaussians_lsq, n_modes, x_data, method):
    x_max = x_data.max()
    x_min = x_data.min()
    gaussians = []

    for i in range(n_modes):
        if ((gaussians_lsq[i]["mean"]>x_max) | (gaussians_lsq[i]["mean"]<x_min)):
            gaussians.append(gaussians_lsq[i])
        else:
            if method=="cluster":
                gaussians.append(gaussians_gmm[i])
            if method=="lsqr":
                gaussians.append(gaussians_lsq[i])

    return gaussians

def fit_multimode(x, y, timestamp, n_modes = None, n_samples = 10000, method="cluster"):
    """
    Fit multimodal Gaussian to aerosol number-size distribution

    Parameters
    ----------

    x : 1d numpy array
        log10 of bin diameters in nm.
    y : 1d numpy array
        Number size distribution
    timestamp : pandas Timestamp
        timestamp associated with the number size distributions
    n_modes : int or `None`
        number of modes to fit, if `None` the number is determined using automatic method
    n_samples : int
        Number of samples to draw from the distribution
        during the fitting process.
    method : str
        `cluster` clustering (gaussian mixture)
        `lsqr` least-squares
    
    Returns
    -------

    dictionary:
        Fit results

    """

    # Convert to pandas Series
    ds = pd.Series(index = x, data = y)

    # Interpolate away the NaN values but do not extrapolate, remove any NaN tails
    s = ds.interpolate(limit_area="inside").dropna()

    # Set negative values to zero
    s[s<0]=0

    # Recover x and y for fitting
    x_interp = s.index.values
    y_interp = s.values

    all_ok = True

    sensitivity = 3

    aic_scores = []

    if len(x_interp)<5:
        all_ok = False
    else:
        coef = np.trapz(y_interp,x_interp)
        samples = af.sample_from_dist(x_interp,y_interp,n_samples)

    if ((n_modes is None) and all_ok):

        windo = int(0.41/np.mean(np.diff(x_interp)))

        y_smooth = savgol_filter(y_interp, window_length=windo, polyorder=1)

        samples_smooth = af.sample_from_dist(x_interp,y_smooth,n_samples)

        n_range = np.arange(1,10)

        aic_scores = Parallel(n_jobs=-1)(delayed(fit_gmm_and_aic)(n, samples_smooth) for n in n_range)

        aic_kneedle = KneeLocator(n_range, 
            aic_scores, curve="convex", 
            direction="decreasing", S=sensitivity)

        if (aic_kneedle.elbow is None):
            print("kneedle was none")
            all_ok = False
        else:           
            n_modes = aic_kneedle.elbow

            # Do the fit using GMM and least squares
            gaussians_gmm = fit_gmm(samples, n_modes, coef)
        
            gaussians_lsq = fit_multimodal_gaussian(x_interp, y_interp, gaussians_gmm)

            if gaussians_lsq is None:
                all_ok = False
                print("LSQ failed")
            else:
                gaussians = find_final_gaussians(gaussians_gmm, gaussians_lsq, n_modes, x_interp, method)

    elif ((n_modes is not None) and all_ok):
        gaussians_gmm = fit_gmm(samples, n_modes, coef)
        gaussians_lsq = fit_multimodal_gaussian(x_interp, y_interp, gaussians_gmm)
        if gaussians_lsq is None:
            print("LSQ failed")
            all_ok = False
        else:
            gaussians = find_final_gaussians(gaussians_gmm, gaussians_lsq, n_modes, x_interp, method)
    else:
        pass

    if all_ok:
        try:
            # Make sure all the data is json compatible
            dp = get_peak_positions(gaussians)
            predicted_ndist = calc_pred(x,gaussians)
            predicted_gaussians = calc_pred_gaussians(x,gaussians)
            total_conc = calc_conc_ndist(x,predicted_ndist)
            mode_concs = calc_conc_gaus(x,gaussians)
            diams = list(x)
            if isinstance(timestamp, str):
                time = timestamp
            else:    
                time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        except:
            dp = []
            gaussians = []
            gaussians_gmm = []
            gaussians_lsq = []
            diams = list(x)
            predicted_ndist = [] 
            predicted_gaussians = []
            total_conc = np.nan
            mode_concs = [np.nan]
            if isinstance(timestamp, str):
                time = timestamp
            else:    
                time = timestamp.strftime("%Y-%m-%d %H:%M:%S")            
    else:
        dp = []
        gaussians = []
        gaussians_gmm = []
        gaussians_lsq = []
        diams = list(x)
        predicted_ndist = [] 
        predicted_gaussians = []
        total_conc = np.nan
        mode_concs = [np.nan]
        if isinstance(timestamp, str):
            time = timestamp
        else:    
            time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            
    # Construct the result dictionary
    result = {
        "time": time,
        "gaussians": gaussians,
        "number_of_gaussians": len(gaussians),
        "peak_diams": dp,
        "predicted_ndist": predicted_ndist,
        "diams": diams,
        "predicted_gauss": predicted_gaussians,
        "total_conc": total_conc,
        "mode_concs": mode_concs,
        "aic_scores":aic_scores,
    }

    return result

def fit_multimodes(df, n_modes = None, n_samples = 10000, method = "cluster"):
    """
    Fit multimodal Gaussian to a aerosol number size distribution (dataframe)

    Parameters
    ----------

    df : pandas DataFrame
        Aerosol number size distribution
    n_modes : int or `None`
        Number of modes to fit, if `None` the number is 
        determined using automatic method for each timestamp
    n_samples : int
        Number of samples to draw from the distribution
        during the fitting process.
    
    Returns
    -------

    list:
        List of fit results
    list:
        List of elapsed times for each fit

    """

    df = df.dropna(how="all",axis=0)

    x = np.log10(df.columns.values.astype(float)*1e9)
    
    fit_results = []
    elapsed_times = []
    
    for j in range(df.shape[0]):
        y = df.iloc[j,:].values.flatten()
        start = time.time()
        fit_result = fit_multimode(x, y, df.index[j], n_modes = n_modes, n_samples = n_samples)
        end = time.time()

        fit_results.append(fit_result)
        elapsed_times.append(end-start)
        
        if (fit_result["number_of_gaussians"]>0):
            print(f'{df.index[j]}: found {fit_result["number_of_gaussians"]} modes in {end-start:.4f} seconds')
        else:
            print(f'{df.index[j]}: found {fit_result["number_of_gaussians"]} modes in {end-start:.4f} seconds')

    return fit_results,elapsed_times
