#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import functions as f
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord 


# In[2]:


print('This library was developed by Dr. F. L. Rommel and finatiated by the Interinstitutional e-Astronomy Laboratory – LIneA (CNPq), Brazil.')
print('Special thanks to the Federal University of Technology – Paraná (UTFPR) for being the host of this pilot project.')
print('Also thanks to Dr. R. C. Boufleur for his great contributions to this project.')
print('Last updated on April 14, 2025.')


# In[3]:


df = pd.DataFrame(pd.read_csv('input_files/input_parameters.csv'))
df


# In[4]:


current_path=os.getcwd()
for b in range(len(df)):
    obj_name = df['Object name'][b]
    if not os.path.exists(obj_name):
        os.mkdir(obj_name)
    #
    if df['download key'][b]=='y':
        f.download_fits_tables(obj_name)
    # Analysis part
    exec("img_"+obj_name+"=[]")
    pp = str(df['path'][b]+'/fits_tables/')
    os.chdir(pp)
    for file in glob.glob("*.fits"):
        exec("img_"+obj_name+".append(file)")
    exec("s = len(img_"+obj_name+")")
    # step 2: obtain the minimum and maximum limits to generate the ephemeris
    exec("data_file = np.array(img_"+obj_name+")")
    filters = np.array([d.split('_')[3] for d in data_file])
    obs_date = np.array([d.split('_')[1][0:8] for d in data_file])
    date_min = min(obs_date)[0:4]+'-'+min(obs_date)[4:6]+'-'+min(obs_date)[6:8]
    date_max = max(obs_date)[0:4]+'-'+max(obs_date)[4:6]+'-'+max(obs_date)[6:8]
    print('***************************************************************************************************')
    print(obj_name, 'directory has a total of', s, 'FITS files acquired between:',date_min,"and",date_max)
    # retrieving the ephemeris from Horizons
    ephem = f.get_ephemeris_horizons(obj_name,date_min,date_max)
    # separating files by filter and searching for the correct PSF measurement
    filter_list = ['zg','zi','zr']
    for item in filter_list:
        images = data_file[filters == item]
        # creating an empty list for each column I want to save in the end of this process
        list_columns = f.list_columns_for_csv()
        for column in list_columns:
            exec(column+' = []')  
        # For each image in the 'imagens' list, do the following:
        for line in tqdm(images):
            resultant_string = f.delta_string(str(df['Delta mag'][b]))   # Create a string with the used delta magnitude to insert in the file's name
            image.append(line)                                            # Save the image name into a list
            hdu_list = fits.open(pp+'/'+line, memmap=True)                # Open the FITS file
            # getting information from the first header of the FITS file
            epoch,cff,cff_err,p_color,mag_zero,mag_zero_sigma = f.get_header_information(hdu_list)
            time_final.append(epoch.jd)                                   # Save the Julian Date into a list
            cf.append(cff)                                                # Save the coefficient color used to the photometric calibrations by ZTF into a list
            cf_err.append(cff_err)                                        # Save the error of the coefficient color into a list
            pcolor.append(p_color)                                        # Save the color used in the photometric calibration by ZTF into a list
            magzero.append(mag_zero)
            magzero_sigma.append(mag_zero_sigma) 
            # Ephemeris interpolation to get predicted RA,DEC position of the target
            parameters = f.ephemeris_interpolation_toimg_time(ephem,epoch)  
            RA = parameters[0]
            DEC = parameters[1]
            predicted_Vmag = parameters[6]
            #
            ra_ephe.append(RA)                                   # Save the apparent RA of the object into a list
            dec_ephe.append(DEC)                                 # Save the apparent DEC of the object into a list
            alpha_final.append(parameters[2])                    # Saves the interpolated alpha into a list
            earth_dist.append(parameters[3])                     # Saves the interpolated delta into a list
            sun_dist.append(parameters[4])                       # Saves the interpolated heliocentric distance into a list
            light_time.append(parameters[5])                     # Save the 1-way light time into a list
            JD_corrected.append(epoch.jd-(parameters[5]/1440))   # Save the JD corrected by 1-way lighttime into a list
            # Calculating the offset between predicted RA,DEC and the positions of each source in the FITS file
            table = Table(hdu_list[1].data)                      # Reads the table available in the second header of the FITS file
            RA2 = np.asarray(table['ra'])                        # Obtain the RA column from the second header
            DEC2= np.asarray(table['dec'])                       # Obtain the DEC column from the second header
            distance = np.sqrt((RA2-RA)**2 + (DEC2-DEC)**2)
            ind = np.argsort(distance)                           # Sort the distances and saves the index order
            mag_i = table['mag'][ind[0:10]]                      # Obtains the instrumental mag of the 10 PSFs closest to the predicted RA,DEC position
            maginst_s = table['sigmag'][ind[0:10]]
            if p_color == 'g-R':
                c = df['color g-R '][b]
                c_err = df['color_g-R_error'][b]
            else:
                c = df['color R-i'][b]
                c_err = df['color R-i error'][b]
            # Calculating the calibrated magnitude using the information provided by the user
            mag_converted = f.maginst_to_magcalibrated(minst=mag_i,minst_err=maginst_s,ZPf=mag_zero,ZPf_err=mag_zero_sigma,
                                                        cf=cff,cf_err=cff_err,color=c,color_err=c_err)# cores vindas do arquivo de input
            # Selecting the closest one that is between the expected range of magnitudes
            magmin = predicted_Vmag-df['Delta mag'][b]                    # Minimum magnitude expected for the target
            magmax = predicted_Vmag+df['Delta mag'][b]                    # Maximum magnitude expected for the target
            filtro_mag = (mag_converted > magmin) & (mag_converted < magmax)  # Filter only the sources inside the expected range 
            if np.sum(filtro_mag) == 0:                                       # if False, the 10 closest sources are not inside the expected range of magnitudes.
                maginst.append(None)                                      
                maginst_sigma.append(None)                                
                mag_new.append(None)
                mag_new_err.append(None)
                ra_source.append(None)
                dec_source.append(None)
                separation.append(None)
                flags.append(None)
                sharp.append(None)
                chi.append(None)
                flaggg.append(False)                                       # Saves a Flag to filter solutions in the next steps of this code
            else:
                idx = np.argwhere(filtro_mag == True)[0][0]                # Selects the first source that attends the expected range of magnitudes
                maginst.append(mag_i[idx])                                 # Saves the instrumental magnitude of the selected source into a list
                maginst_sigma.append(maginst_s[idx])                       # Saves the error of the instrumental magnitude into a list
                mag_new.append(mag_converted[0][idx])                         # Saves the calibrated magnitude of the selected source into a list
                mag_new_err.append(mag_converted[1][idx])                      # Saves the calibrated mag error into a list
                aa = table['ra'][ind[idx]]
                bb = table['dec'][ind[idx]]
                ra_source.append(aa)                                       # Saves the observed RA of the selected source into a list
                dec_source.append(bb)                                      # Saves the observed DEC of the selected source into a list
                ssss = SkyCoord(ra=aa*u.degree, dec=bb*u.degree)
                eeee = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree)
                sep = ssss.separation(eeee)                                # separation of the source from the position given by the ephemerides in module
                sepp = sep.to_value()*3600*1000                            # converts the angle in degrees to milliarcseconds (mas).
                separation.append(sepp)
                ff = table['flags'][ind[idx]]
                flags.append(ff)
                s1 = table['sharp'][ind[idx]]
                sharp.append(s1)
                c1=table['chi'][ind[idx]]
                chi.append(c1)
                flaggg.append(True)
        lista = [image,time_final,cf,cf_err,pcolor,magzero,magzero_sigma,ra_ephe,dec_ephe,alpha_final,earth_dist,sun_dist,
                 light_time,JD_corrected,maginst,maginst_sigma,mag_new,mag_new_err,ra_source,dec_source,separation,flags,sharp,chi,flaggg]
        dd = f.dict_to_csv_file(lista)
        # Created a pandas dataframe from the dictionary created before
        dff = pd.DataFrame(dd)
        # Selects the lines with a flag = True only and drops lines that did not passed in the filters
        dff = dff[dff['flag']==True]
        dff = dff.drop(['flag'], axis=1)
        dff.drop(dff[dff['separation (mas)'] > int(df['separation (mas)'][b])].index, inplace=True)
        f.drop_filters(dff)
        # Making some new columns
        dff['Date (gregorian)'] = pd.to_datetime(dff['Date (JD)'], unit='D', origin='julian')
        dff['Year'] = dff['Date (gregorian)'].dt.year
        dff = dff.drop(['Date (gregorian)'], axis=1)
        dff['Reduced mag'] = f.reduced_magnitude(df = dff)
        # saving the filtered dataframe into a CSV file.
        print(''+obj_name+' was found in',len(dff),'images acquired with filter', item[1])
        dff.to_csv(df['path'][b]+'/'+obj_name+'_'+item[1]+'_horizons_'+resultant_string+'.csv', sep=',', index=False, encoding='utf-8')
        ###########
        PATHH = str(df['path'][b])+'/plots/'
        if not os.path.exists(PATHH): 
            os.makedirs(PATHH)
        if len(dff) > 15:
            if df['yearly_analysis'][b] == 'n':
                f.plot_all_data(obj_name,item[1],dff,path=PATHH)
                f.plot_all_data(obj_name,item[1],dff,path=PATHH,column='Calibrated mag (mag)')
            elif df['yearly_analysis'] == 'y':
                f.plot_single_year(obj_name,item[1],dff,path=PATHH)
                f.plot_single_year(obj_name,item[1],dff,path=PATHH,column='Calibrated mag (mag)')
            # Model 1: All values for phase curve plot
            x = dff['Phase angle (deg)']
            y = dff['Reduced mag']
            # Model 2:Creating the median values for the phase curve plot
            df2 = f.round_by_phase_angle(dff,1)
            x2 = df2['phase_angle_round']
            y2 = df2['Reduced mag']    
            # Doing the phase curve fit and saving the plot
            df_list = [dff,df2]
            x_list =  [x,x2]
            y_list =[y,y2]
            phase_parameters = f.plot_phase_curve_linear(asteroid=obj_name,
                                                       z_filter=item[1],
                                                       path=PATHH,
                                                       df_list=df_list,
                                                       x_list=x_list,
                                                       y_list=y_list)
            # Defining each parameter found from linear fit
            H1 = phase_parameters[0]
            H1_err = phase_parameters[1]
            b1 = phase_parameters[2]
            b1_err = phase_parameters[3]
            H2 = phase_parameters[4]
            H2_err = phase_parameters[5]
            b2 = phase_parameters[6]
            b2_err = phase_parameters[7]
            model = (b2*dff['Phase angle (deg)'])+H2
            ###
            mag = dff['Reduced mag']-model
            for number in range(1,3):
                f.lomb_scargle_fit(asteroid=obj_name,
                                     z_filter=item[1],
                                     df=dff,
                                     corrected_mag=mag,
                                     n_term=number,
                                     f_min=df['cycle_min'][b],
                                     f_max=df['cycle_max'][b],
                                     P_ref=df['P_expected'][b],
                                     path=PATHH)
                
                if df['yearly_analysis'][b] == 'y':
                    for year in dff['Year'].unique():
                        DF = pd.DataFrame(dff.loc[dff['Year']==year])
                        if len(DF) > 30:
                            model_2 = (b2*DF['Phase angle (deg)'])+H2
                            mag_2 = DF['Reduced mag']-model_2
                            
                            f.lomb_scargle_fit(asteroid=obj_name,
                                             z_filter=item[1],
                                             df=DF,
                                             corrected_mag=mag_2,
                                             n_term=number,
                                             f_min=df['cycle_min'][b],
                                             f_max=df['cycle_max'][b],
                                             P_ref=df['P_expected'][b],
                                             path=PATHH,
                                             year=year)   
            
            #### Threshold to perform also the Shevchenko fit to the phase curve.
            if dff['Phase angle (deg)'].max() > 3:
                mask = dff['Phase angle (deg)']<2
                df2 = dff[mask]
                if len(df2) > 5:
                    # Model 1: All values for phase curve plot
                    xx = dff['Phase angle (deg)']
                    yy = dff['Reduced mag']
                    #
                    df11 = pd.DataFrame()
                    df11['alpha'] = xx
                    df11['v'] = yy
                    df11 = df11.sort_values('alpha')
                    df11.insert(0, 'id', df['Object number'][b])
                    # Model 2:Creating the median values for the phase curve plot
                    df22 = f.round_by_phase_angle(dff,0)
                    x22 = df22['phase_angle_round']
                    y22 = df22['Reduced mag']   
                    #
                    df3 = pd.DataFrame()
                    df3['alpha'] = x22
                    df3['v'] = y22
                    df3 = df3.sort_values('alpha')
                    df3.insert(0, 'id', df['Object number'][b])
                    # Doing the phase curve fit and saving the plot
                    df_list1 = [df11,df3]
                    x_list1 =  [xx,x22]
                    y_list1 =[yy,y22]
                    phase_parameters = f.plot_phase_curve_schevchenko(obj_name,
                                                               item[1],
                                                               path=PATHH,
                                                               df_list=df_list1,
                                                               x_list=x_list1,
                                                               y_list=y_list1)
                    # Defining each parameter found from linear fit
                    H1 =  phase_parameters[0]
                    H1_err = phase_parameters[1]
                    a1 =  phase_parameters[2]
                    a1_err = phase_parameters[3]
                    b1 = phase_parameters[4]
                    b1_err = phase_parameters[5]
                    H2 =  phase_parameters[6]
                    H2_err = phase_parameters[7]
                    a2 =  phase_parameters[8]
                    a2_err = phase_parameters[9]
                    b2 = phase_parameters[10]
                    b2_err = phase_parameters[11]
                    #
                    modell = H1-(a1/(1+dff['Phase angle (deg)']))+(b1*dff['Phase angle (deg)'])
                    magg = dff['Reduced mag']-modell
                    for number in range(1,3):
                        f.lomb_scargle_fit(asteroid=obj_name,
                                             z_filter=item[1],
                                             df=dff,
                                             corrected_mag=magg,
                                             n_term=number,
                                             f_min=df['cycle_min'][b],
                                             f_max=df['cycle_max'][b],
                                             P_ref=df['P_expected'][b],
                                             path=PATHH,
                                             shev='yes')
                        if df['yearly_analysis'][b] == 'y':
                            for year in dff['Year'].unique():
                                DF = pd.DataFrame(dff.loc[dff['Year']==year])
                                if len(DF) > 30:
                                    model_2 = (b2*DF['Phase angle (deg)'])+H2
                                    mag_2 = DF['Reduced mag']-model_2
                                    
                                    f.lomb_scargle_fit(asteroid=obj_name,
                                                     z_filter=item[1],
                                                     df=DF,
                                                     corrected_mag=magg,
                                                     n_term=number,
                                                     f_min=df['cycle_min'][b],
                                                     f_max=df['cycle_max'][b],
                                                     P_ref=df['P_expected'][b],
                                                     path=PATHH,
                                                     year=year,
                                                     shev='yes') 

            print('---------------Periodic search ended-------------------------------------------\n\n')
                    


# In[ ]:





# In[ ]:




