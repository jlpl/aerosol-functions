from sklearn.mixture import GaussianMixture
from kneed import KneeLocator
import numpy as np
import pandas as pd
import aerosol.functions as af
from scipy.optimize import curve_fit


def to_meters(x):
    return (10**x)*1e-9

def fit_gmm(samples,n_components,coef,means_init,weights_init):

    if means_init is None and weights_init is None:
        gmm = GaussianMixture(n_components=n_components)
    else:
        gmm = GaussianMixture(
            n_components=n_components,
            means_init=means_init.reshape(-1,1),
            weights_init=weights_init)
    
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

def gaussian(x, amplitude, mean, sigma):
    return amplitude * 1.0/(sigma * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * ((x - mean) / sigma) ** 2.0)

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

def fit_multimodal_gaussian_kde(data_x, data_y, gaussians):
    initial_guesses = []
    lower_bounds = []
    upper_bounds = []

    for g in gaussians:

        initial_guesses.append(g["amplitude"])
        initial_guesses.append(g["mean"]) 
        initial_guesses.append(g["sigma"])

        lower_bounds.append(1)
        lower_bounds.append(-np.inf)
        lower_bounds.append(0.05)        
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
            bounds=(lower_bounds,upper_bounds)
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

def remove_far_peaks(x,gaussians):
    valid_gaussians = []
    for g in gaussians:
        if ((g["mean"]>(x.max()+0.5)) | (g["mean"]<(x.min()-0.5))):
            continue
        else:
            valid_gaussians.append(g)
    return valid_gaussians

def has_too_close_peaks(gaussians, min_distance=0.2):
    peak_positions = np.array([g["mean"] for g in gaussians])
    peak_positions = np.sort(peak_positions)  # Sort the numbers
    peak_position_differences = np.diff(peak_positions)  # Compute consecutive differences
    return np.any(peak_position_differences < min_distance)

def fit_multimode(x, y, timestamp, n_modes = None, n_samples = 10000):
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
    
    Returns
    -------

    `dict`:
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

    # There should be at least 3 points available
    if len(x_interp)<3:
        all_ok = False
    else:
        coef = np.trapz(y_interp,x_interp)
        samples = af.sample_from_dist(x_interp,y_interp,n_samples)

    if ((n_modes is None) and all_ok):

        scores = []
        n_range = []
        for n in range(1,9):
            gaussians_gmm = fit_gmm(samples, n, coef, None, None)
            gaussians_lsq = fit_multimodal_gaussian_kde(x_interp, y_interp, gaussians_gmm)
            
            if ((gaussians_lsq is None) & (len(scores)==0)):
                continue
            elif gaussians_lsq is None:
                scores.append(scores[-1])
                n_range.append(n)
            else:
                pred = calc_pred(x_interp,gaussians_lsq)
                score = mse(pred,y_interp)
                if len(scores)>0:
                    if score > scores[-1]:
                        scores.append(scores[-1])
                    else:
                        scores.append(score)
                else:
                    scores.append(score)
                n_range.append(n)

        sensitivity = 3
        
        kneedle = KneeLocator(n_range, scores, curve="convex", direction="decreasing", S=sensitivity)

        if kneedle.elbow is None:
            print("kneedle was none")
            all_ok = False

        else:
                
            n_modes = kneedle.elbow
            # Do the fit using GMM and least squares
            gaussians_gmm = fit_gmm(samples, n_modes, coef, None, None)
        
            # use it as initial guess for least squares fit
            gaussians_lsq = fit_multimodal_gaussian_kde(x_interp, y_interp, gaussians_gmm)

            if gaussians_lsq is None:
                print("final fit was none")
                all_ok = False
            else:  
              while True:
                  if has_too_close_peaks(gaussians_lsq):
                      n_modes = n_modes-1
                      gaussians_gmm = fit_gmm(samples, n_modes, coef, None, None)
                      gaussians_lsq = fit_multimodal_gaussian_kde(x_interp, y_interp, gaussians_gmm)
                      if gaussians_lsq is None:
                          break
                  else:
                      break

              if gaussians_lsq is not None:
                  gaussians = remove_far_peaks(x_interp,gaussians_lsq)
                  if len(gaussians)==0:
                      print("final number of gaussians was zero")
                      all_ok = False
              else:
                  print("final fit was none")
                  all_ok = False
    elif ((n_modes is not None) and all_ok):
        gaussians_gmm = fit_gmm(samples, n_modes, coef, None, None)
        gaussians = fit_multimodal_gaussian_kde(x_interp, y_interp, gaussians_gmm)
        if gaussians is None:
            print("fit was none")
            all_ok = False
    else:
        pass

    if all_ok:
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
    else:
        dp = []
        gaussians = []
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
    }

    return result

def fit_multimodes(df, n_modes = None, n_samples = 10000):
    """
    Fit multimodal Gaussian to a aerosol number size distribution (dataframe)

    Parameters
    ----------

    df : pandas DataFrame
        Aerosol number size distribution
    n_modes : int or `None`
        number of modes to fit, if `None` the number is determined using automatic method for each timestamp
    n_samples : int
        Number of samples to draw from the distribution
        during the fitting process.
    
    Returns
    -------

    list:
        List of fit results

    """
    # Remove all nan rows
    df = df.dropna(how="all",axis=0)

    x = np.log10(df.columns.values.astype(float)*1e9)
    fit_results = []
    for j in range(df.shape[0]):
        y = df.iloc[j,:].values.flatten()
        fit_result = fit_multimode(x, y, df.index[j], n_modes = n_modes, n_samples = n_samples)
        
        fit_results.append(fit_result)
        
        if (fit_result["number_of_gaussians"]>0):
            print(f'{df.index[j]}: found {fit_result["number_of_gaussians"]} modes')
        else:
            print(f'{df.index[j]}: found {fit_result["number_of_gaussians"]} modes')

    return fit_results

