import pyedra
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.timeseries import LombScargle
from scipy.interpolate import CubicSpline
from astroquery.jplhorizons import Horizons
############################################################################
def build_download_links(csv_file, path_to_save_fits):
    strings = []
    for i in range(len(csv_file)):
        if len(str(csv_file['field'][i])) == 6:
            n = str(csv_file['field'][i])
        elif len(str(csv_file['field'][i])) == 3:
            n = '000'+str(csv_file['field'][i])
        else:
            n = '00'+str(csv_file['field'][i])
        if len(str(csv_file['ccdid'][i]))==1:
            ccdid = '0'+str(csv_file['ccdid'][i])
        else:
            ccdid =str(csv_file['ccdid'][i])
        filefracday=str(csv_file['filefracday'][i])
        a = 'wget -P '+path_to_save_fits+' https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/'+filefracday[0:4]+'/'+filefracday[4:6]+filefracday[6:8]+'/'+filefracday[8:]+'/'+'ztf_'+str(csv_file['filefracday'][i])+'_'+n+'_'+str(csv_file['filtercode'][i])+'_c'+ccdid+'_'+str(csv_file['imgtypecode'][i])+'_q'+str(csv_file['qid'][i])+'_psfcat.fits '
        strings.append(a)   
    np.savetxt('psfcat_links_to_download.sh', strings, fmt="%s")
############################################################################
def delta_string(mag_delta):
    # code from https://www.scaler.com/topics/replace-a-character-in-a-string-python/
    original_string = mag_delta
    # ``.`` has to be replaced with a different character
    replaced_character = "."
    # ``_`` will replace ``d`
    new_character = "_"
    # an empty string to store the result
    resultant_string = ""
    for character in original_string:
        if character == replaced_character:
            resultant_string += new_character
        else:
            resultant_string += character
    return resultant_string
############################################################################
def dict_to_csv_file(list_of_variables):
    dd = {
        'Name':list_of_variables[0],#
        'Date (JD)':list_of_variables[1],#
        '1-way-light-time (min)':list(list_of_variables[12]),#
        'Date_corrected (JD)':list_of_variables[13],#
        'ephemeris RA (deg)':list_of_variables[7],#
        'ephemeris DEC (deg)':list_of_variables[8],#
        'Phase angle (deg)':list_of_variables[9],#
        'Heliocentric dist. (au)':list_of_variables[11],#
        'Geocenter dist. (au)':list_of_variables[10],#
        'source RA (deg)':list_of_variables[18],#
        'source DEC (deg)':list_of_variables[19],#
        'separation (mas)':list_of_variables[20],#
        'ZP (mag)':list_of_variables[5],#
        'ZP error (mag)':list_of_variables[6],#
        'Instrumental mag (mag)':list_of_variables[14],#
        'Instrumental mag error (mag)':list_of_variables[15],#
        'Calibrated mag (mag)':list_of_variables[16],#
        'Calibrated mag error (mag)':list_of_variables[17],#
        'Color coefficient':list_of_variables[2], #
        'Color coefficient uncertainty':list_of_variables[3], #
        'Pcolor':list_of_variables[4],#
        'Chi':list_of_variables[23],#
        'Sharp':list_of_variables[22],#
        'flags':list_of_variables[21],#
        'flag':list_of_variables[24]#
        }
    return dd
############################################################################
def drop_filters(df, mag_n = 4.0, sep_n = 4.0):
    '''
    This function aims to drop measurements that did not respect some quality flags presented in ZTF documentation.
    Also, it deleates points based on the calibrated magnitude error bars (default = 4x the median error bar) and
    poits that have an astrometric separation greater than a given value in milliarcseconds (mas) (default = 4x the median separation).
    
    Parameters
    ----------
    df: pandas DataFrame
    Provide the name of the pandas dataframe to be used.
    
    mag_n: int,float (optional)
    Provide a number to be multiplied by the median error in magnitude and stablish an acceptable upper limit for this value on your data.
    Default: 4.0
    
    sep_n: int (optional)
    Provide a number to be multiplied by the median separation (mas) and stablish an acceptable upper limit for this value on your data set.
    Default: 4.0
    ''' 
    median_err = np.median(df['Calibrated mag error (mag)'])
    median_sep = np.median(df['separation (mas)'])
    df.drop(df[df['flags'] != 0].index, inplace=True) # deleate rows with flags different from zero (indicating a masked pixed near the the source).
    df.drop(df[df['Sharp'] > 50].index, inplace=True) # deleate rows with not PSF like formats (see documentation page 81)
    df.drop(df[df['Sharp'] < -50].index, inplace=True) # deleate rows with not PSF like formats (see documentation page 81)
    df.drop(df[df['Chi'] > 5].index, inplace=True) # deleate rows with ratio of the (RMS in PSF-fit residuals / expected RMS from uncertainties) greater than 5 
    df.drop(df[df['Calibrated mag error (mag)'] > float(median_err*mag_n)].index, inplace=True)
    df.drop(df[df['separation (mas)'] > float(median_sep*sep_n)].index, inplace=True)
############################################################################ 
def drop_years(df,year_list):
    '''
    Function wrote to deleate data points acquired in some specficic year, defined by the user in a initial list.
 
    Parameters
    ----------
    df: pandas DataFrame
    Provide the name of the pandas dataframe to be used.
    
    year_list: list
    The list with all years you want to exclude from the data set.
    Years must be given as integer numbers.
    '''
    if len(year_list) == 1:
        if type(year_list[0]) == float:
            print('Year must be given as an integer value')
        else:
            df.drop(df[df['Year'] == year_list[0]].index, inplace=True)
    elif len(year_list) > 1:
        for i in year_list:
            if type(i) == float:
                print('Year must be given as an integer value')
            else:
                df.drop(df[df['Year'] == i].index, inplace=True)
    else:
        print('Input list is empty')  

def ephemeris_interpolation_toimg_time(ephem,epoch):
    # epoch = Time(epoch)
    cs_RA = CubicSpline(ephem['datetime_jd'].data, ephem['RA'].data)
    predicted_RA = cs_RA(epoch.jd)
    cs_DEC = CubicSpline(ephem['datetime_jd'].data, ephem['DEC'].data)
    predicted_DEC = cs_DEC(epoch.jd)
    cs_alpha = CubicSpline(ephem['datetime_jd'].data, ephem['alpha'].data)
    predicted_alpha = np.round(cs_alpha(epoch.jd),decimals=2)
    cs_delta = CubicSpline(ephem['datetime_jd'].data, ephem['delta'].data)
    predicted_delta = cs_delta(epoch.jd)
    cs_heldist = CubicSpline(ephem['datetime_jd'].data, ephem['r'].data)
    predicted_heldist = cs_heldist(epoch.jd)
    cs_lighttime = CubicSpline(ephem['datetime_jd'].data, ephem['lighttime'].data)
    predicted_lighttime = cs_lighttime(epoch.jd)
    cs_Vmag = CubicSpline(ephem['datetime_jd'].data, ephem['V'].data)
    predicted_Vmag = cs_Vmag(epoch.jd)
    return [predicted_RA,predicted_DEC,predicted_alpha,predicted_delta,predicted_heldist,predicted_lighttime,predicted_Vmag]
############################################################################
def file_split_name(file,position):
    '''
    This function aims to split an string that corresponds to a file name and select one part of it.
    
    Parameters
    ----------
    file: str
    The string to be splited.
    
    position: int
    The position of the partial string after spliting it.
    
    Returns
    -------
    A string corresponding to the part of interest after spliting the initial string.
    '''
    return file.split('/')[8].split('_')[position]
############################################################################
def get_ephemeris_horizons(target,start_date,end_date,step='4h',id_type='smallbody'):
    '''
    This function retrieves the JPL ephemeris for a given object.
    
    Parameters
    ----------
    target: str or int
    The name or id of the object of interest.
    
    start_date/end_date: str with the format YYYY-MM-DD
    The date to start and end the ephemeris downloaded from JPL. 
    
    step: str
    The step of the positions get from JPL. 
    Default: '4h'    
    
    id_type: str
    The type of information you provided for id parameter according to astroquery.jplhorizons documentation.
    Default: 'smallbody'
    '''
    obj = Horizons(id=target, location='I41',epochs={'start': start_date,'stop': end_date,'step': step},id_type=id_type)
    ephem=obj.ephemerides(extra_precision=True)
    return ephem
############################################################################
def get_header_information(header_list):
    '''
    This function intends to get some information from the FITS files headers.
    '''
    hdr = header_list[0].header                          # Open the first header of the FITS file
    epoch = Time(hdr['OBSMJD'],format='mjd')             # Date of the image converted to julian date (JD)
    cff = hdr['CLRCOEFF']                                # Coefficient color used to the photometric calibrations by ZTF
    cff_err = hdr['CLRCOUNC']                            # Uncertainty of the coefficient color used to the photometric calibrations by ZTF
    pcolor = hdr['PCOLOR']                               # Get the color used in the photometric calibration by ZTF into a list
    mag_zero = hdr['MAGZP']                              # Obtains and saves into a list the image zero point
    mag_zero_sigma = hdr['MAGZPUNC']                     # Obtains the error of the image zero point ans saves into a list
    return [epoch,cff,cff_err,pcolor,mag_zero,mag_zero_sigma]
############################################################################
def linear_absolute_mag_fit(df,x,y):
    '''
    Function to fit a linear function to the phase curve.
    The absolute magnitude and slope (beta) are returned. 
    Also, the function returns the coeficients errors from the diagonal of the covariance matrix (probably overestimated).
    
    Parameters
    ----------
    df: pandas DataFrame
        Provide the name of the pandas dataframe to be used.
        
    x: pandas.core.series.Series
        Phase angles of each observation.
        
    y: pandas.core.series.Series
        Calibrated magnitude corrected by heliocentric and geocentric distances.
        
    Returns
    -------
    Both coeficients (H, beta) from the linear fit and the error bars calculated from the covariance matrix.
    
    '''    
    m, b = np.polyfit(x, y, 1, rcond=None, full=False, w=None, cov=True)
    A = round(np.sqrt(b[0][0]),3)
    B = round(np.sqrt(b[1][1]),3)
    beta=round(m[0],3)
    H = round(m[1],3)
    result=H,B,beta,A
    return result
############################################################################
def list_columns_for_csv():
    '''
    The aim of this function is only provide a list of columns to save in the final CSV per filter.
    '''
    list_colums = ['image',
                   'time_final',
                   'cf',
                   'cf_err',
                   'pcolor',
                   'magzero',
                   'magzero_sigma',
                   'ra_ephe',
                   'dec_ephe',
                   'alpha_final',
                   'earth_dist',
                   'sun_dist',                   
                   'light_time',
                   'JD_corrected',
                   'maginst',
                   'maginst_sigma',
                   'mag_new',
                   'mag_new_err',
                   'ra_source',
                   'dec_source',
                   'separation',
                   'flags',
                   'sharp',
                   'chi',
                   'flaggg'] 
    return list_colums
#####################################################################
def lomb_scargle_fit(asteroid,z_filter,df,corrected_mag,path,n_term=1,f_min=0.5,f_max=12,P_ref=8,year='all',P_dec=5,shev='No'):
    '''
    This function is aimed to derive the strongest frequency on the data using the lomb-scargle method.
    Also it makes the periodogram plots.
    
    Parameters
    ----------
    asteroid: str
    The name of the asteroid to print in the plot's title.
    
    z_filter: str
    The name of filter used in the data set.
    
    df: pandas DataFrame
    Provide the name of the pandas dataframe to be used.
    
    corrected_mag: pandas.core.series.Series
    The object's reduced magnitude corrected by the solar phase angle effects.
    
    path: str
    The path to save the plots.
    
    n_term: int
    The number of terms to be used in the lomb-scargle fit (default=1).
    
    f_min:float
    The minimum frequency (cycles/day) to be used in the search (default=0.5).
    
    f_max:float
    The maximum frequency (cycles/day) to be used in the search (default=12.0).
    
    P_ref: float
    The rotational period (published) for reference in the periodogram plot.
    If not provided the function will use P_ref = 8h as default.
    
    year: int
    The year that data were acquired (default = 'all').
    
    P_dec: int
    The number of decimals to be presented in the plots (default=5).
    '''
    FSS = 20
    fig_width=8
    fig_height=4
    #
    JD = df['Date_corrected (JD)']
    ls = LombScargle(JD, corrected_mag, nterms=n_term)
    frequencia, potencia = ls.autopower(minimum_frequency=float(f_min), maximum_frequency=float(f_max))
    best_freq = frequencia[np.argmax(potencia)]
    ####
    fig = plt.figure(figsize=(fig_width,fig_height))
    plt.plot(frequencia, potencia,color='c')
    plt.title(asteroid+' - '+z_filter+' - '+str(n_term)+' terms - '+str(year),fontsize=FSS)
    plt.xlabel('Frequency (cycles/day)',fontsize=FSS)
    plt.ylabel('Power LS',fontsize=FSS)
    plt.axvline((24/P_ref),color='k',linestyle=':',lw=2,label='Publ = '+str(P_ref)+' h')
    plt.plot(best_freq,potencia.max(),'ro',label='Peak = {:.2f} h'.format(24/best_freq))
    if n_term == 1:
        probabilities = [0.001]
        false_alarm = ls.false_alarm_level(probabilities)
        plt.plot([frequencia.min(), frequencia.max()],[false_alarm, false_alarm], 'r--', label='99% confidence')
    plt.xticks(fontsize=FSS)
    plt.yticks(fontsize=FSS)
    plt.legend(fontsize=FSS-5,loc='upper left')# , bbox_to_anchor=(1, 1)
    plt.tight_layout()
    if shev=='yes':
        plt.savefig(path+asteroid+'_'+z_filter+'_'+str(year)+'_term'+str(n_term)+'_periodogram_Shevchenko.jpg',format='jpg')
    else:
        plt.savefig(path+asteroid+'_'+z_filter+'_'+str(year)+'_term'+str(n_term)+'_periodogram.jpg',format='jpg')
    plt.close()
    #####
    # lets create a variable name 'fase_orbital' with 2 times de period find before
    fase = (JD*best_freq) % 1
    fase_orbital, mag_orbital = np.concatenate([fase,fase+0]), np.concatenate([corrected_mag,corrected_mag])
    t_fit = np.linspace(0, 1/best_freq, 1000)
    y_fit = ls.model(t_fit, best_freq)
    phase_fit = (t_fit*best_freq) % 1
    model = ls.model(JD, best_freq)
    std = np.std(corrected_mag-model,ddof=1)
    amp = y_fit.max()-y_fit.min()
    P = np.round(24/best_freq,decimals=P_dec)
    #
    fig = plt.figure(figsize=(fig_width,fig_height))
    plt.title(asteroid+' - '+z_filter+' - '+str(year)+' - '+str(len(df))+' points - P ='+str(P)+' hours',fontsize=FSS)
    plt.plot(fase_orbital, mag_orbital, '.')
    plt.plot((t_fit*best_freq) % 1, y_fit, 'r.', label='LS best model')
    plt.xlabel('Frequency (cicles/day)',fontsize=FSS)
    plt.ylabel('Mag',fontsize=FSS)
    plt.annotate('std = {:.2f} mag'.format(std),(0.7,0.9),xycoords='axes fraction',fontsize=FSS-3)
    plt.annotate(r'$\Delta$m = {:.2f} mag'.format(amp),(0.7,0.80),xycoords='axes fraction',fontsize=FSS-3)
    plt.gca().invert_yaxis()
    plt.xticks(fontsize=FSS)
    plt.yticks(fontsize=FSS)
    plt.tight_layout()
    if shev=='yes':
        plt.savefig(path+asteroid+'_'+z_filter+'_'+str(year)+'_term'+str(n_term)+'_phased_curve_Shevchenko.jpg',format='jpg')
    else:
        plt.savefig(path+asteroid+'_'+z_filter+'_'+str(year)+'_term'+str(n_term)+'_phased_curve.jpg',format='jpg')
    plt.close()
##############################################################################################################
# This function was based in Eq. 2 from ztf_pipelines_deliverables.pdf file. Last updated in November 20, 2023.
def maginst_to_magcalibrated(minst,minst_err,ZPf,ZPf_err,cf,cf_err,color,color_err):
    """ 
    Calculate the calibrated magnitude with uncertainties for a source measured by the PSF photometry made by ZTF.
    
    Parameters
    ----------
    minst: float
        Instrumental magnitude as measured by ZTF pipeline.
        
    minst_err: float
        The uncertainty of the instrumental magnitude as measured by ZTF pipeline.
        
    ZPf: float
        Photometric zero-point of the image used to the photometric calibration by ZTF pipeline.
        
    ZPf_err: float
        The uncertainty of the zero-point of the image used to photometric calibration.
        
    cf: float
        Color coefficient used by ZTF pipeline to calibrate the image.
        
    cf_err: float
        The uncertainty of the color coefficient used by ZTF pipeline to calibrate the image.
    
    Pcolor: str
        The Pcolor used by ZTF pipeline to calibrate the images before photometry.
        Usually g-R for images on g and r filters. R-i for images on i filter.
        
    color: float
        Target's g-r color index obtained from literature.
        Put zero if this information is not available.
        
    color_err: float
        Uncertainty of the target's g-r color index obtained from literature.
        Put zero if this information is not available.
        
    Returns
    -------
    mcal: float
        Calibrated magnitude (mag).
        
    mcal_err: float
        The uncertainty of the calibrated magnitude (mag).
    """
    mcal=minst+ZPf+(cf*color)
    #
    A = minst_err**2
    B = ZPf_err**2
    C = (color*cf_err)**2
    D = (cf*color_err)**2
    mcal_err = np.sqrt(A+B+C+D)
    #
    return mcal,mcal_err
##########################################################
def plot_all_data(asteroid,z_filter,df,path,column='Reduced mag'):
    tmp_str = column.split(' ')[0]
    fig_width = 1500
    fig_height = 500
    
    fig=px.scatter(df,
                   x='Date_corrected (JD)',
                   y=column,
                   error_y='Calibrated mag error (mag)',
                   color='separation (mas)') 
        
    fig.update_layout(title={'text': str(asteroid)+' - '+z_filter+' filter - all - '+str(len(df))+' points - mag_err: '+str(round(np.median(df['Calibrated mag error (mag)']),2)),
                                 'y':0.95,'x':0.5},
                         font=dict(
                                family="serif",
                                size=30,
                                color="Black"
                            ))
    fig.update_yaxes(automargin=True,autorange="reversed")
    fig.write_image(path+asteroid+'_'+z_filter+'_all_general_'+tmp_str+'_mag.jpg',
                    width=fig_width,
                    height=fig_height)
    fig.data = []
#################################################################################################################
def plot_Hmag_curve(asteroid,z_filter,df,path,beta):
    H_mag = df['Reduced mag']-(beta*df['Phase angle (deg)'])
    df['H mag'] = H_mag
    
    # tmp_str = column.split(' ')[0]
    fig_width = 1500
    fig_height = 500
    
    fig=px.scatter(df,
                   x='Date_corrected (JD)',
                   y='H mag',
                   error_y='Calibrated mag error (mag)',
                   color='separation (mas)') 
        
    fig.update_layout(title={'text': str(asteroid)+' - '+z_filter+' filter - all - '+str(len(df))+' points - mag_err: '+str(round(np.median(df['Calibrated mag error (mag)']),2)),
                                 'y':0.95,'x':0.5},
                         font=dict(
                                family="serif",
                                size=30,
                                color="Black"
                            ))
    fig.update_yaxes(automargin=True,autorange="reversed")
    fig.write_image(path+asteroid+'_'+z_filter+'_all_general_H_mag.jpg',
                    width=fig_width,
                    height=fig_height)
    fig.data = []   
#################################################################################################################
def plot_phase_curve_linear(asteroid,z_filter,path,df_list,x_list,y_list):
    '''
    This function creates the plot of the phase curve and fits a linear function to them.
    It also may calculate the median of the points and fit a linear function to them as well.
    
    Requires parameters
    -------------------
    asteroid: str
    The name of the asteroid to write in the title
    
    z_filter: str
    The filter used in the current data set
    
    path: str
    The complete path to save the plot
    
    df_list: list of pandas Dataframes
    Provide a list with the names of the pandas dataframes to be used.
    
    x_list,y_list = list of pandas.core.series.Series
    Provide phase angles and reduced magnitudes values to be used.
    
    Returns
    -------
    It saves a phase curve plot into the provided path.
    '''
    FSS = 20
    fig_width=8
    fig_height=4
    
    #
    if len(df_list) > 2:
        print('Only 2 dataframes can be used in the plot')
    #
    H,H_err,beta,beta_err = linear_absolute_mag_fit(df_list[0],x_list[0],y_list[0])
    model1 = beta*x_list[0] + H
    text_to_anotate_1 = r'$\beta_1$:   {:.3f} +/- {:.3f}'.format(beta,beta_err)
    text_to_anotate_2 = r'$H_1$:   {:.3f} +/- {:.3f}'.format(H,H_err)
    #
    fig = plt.figure(figsize=(fig_width,fig_height))
    plt.title(asteroid+' - '+z_filter+' filter - '+str(len(df_list[0]))+' points',fontsize=FSS)
    plt.plot(x_list[0], y_list[0], '.',ms=10,color='grey', label='All (model 1)')
    plt.plot(x_list[0], model1, '-',color='grey',lw=3)
    #
    if len(df_list)>1:
        H2,H_err2,beta2,beta_err2 = linear_absolute_mag_fit(df_list[1],x_list[1],y_list[1])
        model2 = beta2*x_list[1] + H2
        text_to_anotate_3 = r'$\beta_2$:   {:.3f} +/- {:.3f}'.format(beta2,beta_err2)
        text_to_anotate_4 = r'$H_2$:   {:.3f} +/- {:.3f}'.format(H2,H_err2)
        #
        plt.plot(x_list[1], y_list[1], 'v',ms=7,color='r', label='Median (model 2)')
        plt.plot(x_list[1], model2, 'r--', lw=2)
        plt.annotate(text_to_anotate_3,(0.55,0.15),xycoords='axes fraction',fontsize=FSS-3)
        plt.annotate(text_to_anotate_4,(0.55,0.05),xycoords='axes fraction',fontsize=FSS-3)
    #
    plt.annotate(text_to_anotate_1,(0.05,0.15),xycoords='axes fraction',fontsize=FSS-3)
    plt.annotate(text_to_anotate_2,(0.05,0.05),xycoords='axes fraction',fontsize=FSS-3)    
    plt.xlabel('Phase angle (deg)',fontsize=FSS)
    plt.ylabel('Reduced mag (mag)',fontsize=FSS)
    plt.xticks(fontsize=FSS)
    plt.yticks(fontsize=FSS)
    plt.ylim(y_list[0].min()-0.55,y_list[0].max()+0.55)
    plt.gca().invert_yaxis()
    plt.legend(fontsize=FSS-5,loc=1)
    plt.tight_layout()
    plt.savefig(path+asteroid+'_'+z_filter+'_phase_curve_linear.jpg',format='jpg')
    plt.close()
    #
    if len(df_list) == 1:
        result = [H,H_err,beta,beta_err]
    else:
        result = [H,H_err,beta,beta_err,H2,H_err2,beta2,beta_err2]
    return result
#################################################################################################################
def plot_phase_curve_schevchenko(asteroid,z_filter,path,df_list,x_list,y_list):
    '''
    This function creates the plot of the phase curve and fits the schevchenko model to them.
    It also may calculate the median of the points and fit a linear function to them as well.
    
    Requires parameters
    -------------------
    asteroid: str
    The name of the asteroid to write in the title
    
    z_filter: str
    The filter used in the current data set
    
    path: str
    The complete path to save the plot
    
    df_list: list of pandas Dataframes
    Provide a list with the names of the pandas dataframes to be used.
    
    x_list,y_list = list of pandas.core.series.Series
    Provide phase angles and reduced magnitudes values to be used.
    
    Returns
    -------
    It saves a phase curve plot into the provided path.
    '''
    FSS = 20
    fig_width=8
    fig_height=4
    
    #
    if len(df_list) > 2:
        print('Only 2 dataframes can be used in the plot')
    #
    H,H_err,a,a_err,b,b_err = schevchenko_absolute_mag(df_list[0],x_list[0], y_list[0])
    model1 = H-(a/(1+x_list[0]))+(b*x_list[0])
    text_to_anotate_1 = r'$H_1$:   {:.3f} +/- {:.3f}'.format(H,H_err)
    text_to_anotate_2 = r'$a_1$:   {:.3f} +/- {:.3f}'.format(a,a_err)
    text_to_anotate_3 = r'$b_1$:   {:.3f} +/- {:.3f}'.format(b,b_err)
    #
    fig = plt.figure(figsize=(fig_width,fig_height))
    plt.title(asteroid+' - '+z_filter+' filter - '+str(len(df_list[0]))+' points',fontsize=FSS)
    plt.plot(x_list[0], y_list[0], '.',ms=10,color='skyblue', label='All (model 1)')
    plt.plot(x_list[0], model1, '.',ms=10,color='blue')
    #
    if len(df_list)>1:
        H2,H_err2,a2,a_err2,b2,b_err2 = schevchenko_absolute_mag(df_list[1],x_list[1], y_list[1])
        model2 = H2-(a2/(1+x_list[1]))+(b2*x_list[1])
        text_to_anotate_4 = r'$H_2$:   {:.3f} +/- {:.3f}'.format(H2,H_err2)
        text_to_anotate_5 = r'$a_2$:   {:.3f} +/- {:.3f}'.format(a2,a_err2)
        text_to_anotate_6 = r'$b_2$:   {:.3f} +/- {:.3f}'.format(b2,b_err2)
        #
        plt.plot(x_list[1], y_list[1], '.',ms=7,color='r', label='Median (model 2)')
        plt.plot(x_list[1], model2, 'r--', lw=2)
        plt.annotate(text_to_anotate_4,(0.55,0.23),xycoords='axes fraction',fontsize=FSS-3)
        plt.annotate(text_to_anotate_5,(0.55,0.13),xycoords='axes fraction',fontsize=FSS-3)
        plt.annotate(text_to_anotate_6,(0.55,0.03),xycoords='axes fraction',fontsize=FSS-3)
        #
    plt.annotate(text_to_anotate_1,(0.05,0.23),xycoords='axes fraction',fontsize=FSS-3)
    plt.annotate(text_to_anotate_2,(0.05,0.13),xycoords='axes fraction',fontsize=FSS-3)
    plt.annotate(text_to_anotate_3,(0.05,0.03),xycoords='axes fraction',fontsize=FSS-3)    
    plt.xlabel('Phase angle (deg)',fontsize=FSS)
    plt.ylabel('Reduced mag (mag)',fontsize=FSS)
    plt.xticks(fontsize=FSS)
    plt.yticks(fontsize=FSS)
    plt.ylim(y_list[0].min()-0.55,y_list[0].max()+0.55)
    plt.gca().invert_yaxis()
    plt.legend(fontsize=FSS-5,loc=1)
    plt.tight_layout()
    plt.savefig(path+asteroid+'_'+z_filter+'_phase_curve_schevchenko.jpg',format='jpg')
    plt.close()
    #
    if len(df_list) == 1:
        result = [H,H_err,a,a_err,b,b_err]
    else:
        result = [H,H_err,a,a_err,b,b_err,H2,H_err2,a2,a_err2,b2,b_err2]
    return result
##########################################################
def plot_single_year(asteroid,z_filter,df,path,beta):
    H_mag = df['Reduced mag']-(beta*df['Phase angle (deg)'])
    df['H mag'] = H_mag
    
    # tmp_str = column.split(' ')[0]
    fig_width = 1500
    fig_height = 500
    for item in df["Year"].unique():
        DF = pd.DataFrame(df.loc[df['Year'] == item])
        fig=px.scatter(DF,
                       x='Date_corrected (JD)',
                       y='H mag',
                       error_y='Calibrated mag error (mag)',
                       color='separation (mas)') 
        
        fig.update_layout(title={'text': str(asteroid)+' - '+z_filter+' filter - '+str(item)+' - '+str(len(DF))+' points - mag_err: '+str(round(np.median(DF['Calibrated mag error (mag)']),2)),
                                 'y':0.95,'x':0.5},
                         font=dict(
                                family="serif",
                                size=30,
                                color="Black"
                            ))
        fig.update_yaxes(automargin=True,autorange="reversed")
        fig.write_image(path+asteroid+'_'+z_filter+'_'+str(item)+'_general_H_mag.jpg',
                        width=fig_width,
                        height=fig_height)
        fig.data = []

#############################
def reduced_magnitude(df):
    '''
    Function to calculate the magnitude of the object corrected by Earth and Sun's distances.
    Add a column in the current data frame named Reduced_mag (mag).
    
    Parameters
    ----------
    df: pandas DataFrame
    Provide the name of the pandas dataframe to be used.

    '''
    return df['Calibrated mag (mag)'] - 5*np.log10(pd.to_numeric(df['Geocenter dist. (au)']*df['Heliocentric dist. (au)']))
########################################################    
def round_by_phase_angle(df,decimals=0):
    '''
    This function aims to round reduced magnitude values by using the phase angle value.
    
    Parameters
    ----------
    df: pandas DataFrame
    Provide the name of the pandas dataframe to be used.

    decimals: int (optional)
    The number of decimals to be used to round values.
    If not provided the function will round to the unity, i.e, 1.3 will be round to 1.
    Default = 0.
    
    Returns
    -------
    df2: pandas Dataframe
    Creates a second dataframe with two columns (phase angle and reduced magnitude)
    
    '''
    
    df['phase_angle_round'] = df['Phase angle (deg)'].round(decimals=decimals)
    A = df.groupby(['phase_angle_round'])['Reduced mag'].median()
    df2 = A.reset_index()
    return df2
##############################################################
def schevchenko_absolute_mag(df,x,y):
    '''
    Function to fit a Shevchenko model to the phase curve, which considers the oposition effect.
    The absolute magnitude (H), the oposition amplitude (a) and the error in H (b) are returned. 
    Also, the function returns the coeficients errors from the covariance matrix (probably overestimated).
    
    Parameters
    ----------
    df: pandas dataframe
        Provide the name of the pandas dataframe containing only three columns named exactly as 'id', 'alpha', 'v'.
        The 'id' is the asteroid identification number, the 'alpha' is the phase angle of the measurement, and
        'v' in this application, is the reduced magnitude in a given ZTF filter.
    
    x: pandas.core.series.Series
        Phase angles of each observation.
        
    y: pandas.core.series.Series
        Calibrated magnitude corrected by heliocentric and geocentric distances.
    
    Returns
    -------
    The tree coeficients proposed by Schevchenko, where H is the absolute magnitude with uncertainties (H_err) calculated by linear
    extrapolation to zero; 'a' is the fit parameter with uncertainties (a_err) characterizing the opposition efect amplitude and;
    'b' is the fit parameter with uncertainties (b_err) describing the linear part of the phase curve dependence (usually greater 
    phase angle values).
    
    '''
    if len(df.columns) > 3:
        print("Please provide a dataframe with only three columns ('id','alpha','v')")
    else:
        HG = pyedra.Shev_fit(df)
        H = round(HG.V_lin[0],2)
        H_err = round(HG.error_V_lin[0],2)
        a = round(HG.b[0],2)
        a_err = round(HG.error_b[0],2)
        b = round(HG.c[0],2)
        b_err = round(HG.error_b[0],2)
        
        return H,H_err,a,a_err,b,b_err
