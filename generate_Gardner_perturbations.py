#!/usr/bin/env python3

# Quentin Brissaud and Antoine Turquet, NORSAR, July 2021
# Adapted from Peter Nasholm Matlab code for Gardner perturbations is translated to Python  

# Below are his notes from the original code  
# % Sven Peter Nasholm, NORSAR, March 2015
# %
# % Generate perturbation using Gardner's spectrum from the paper:
# %    JOURNAL OF GEOPHYSICAL RESEARCH, VOL. 98, NO. D1, PAGES 1035-1049, JANUARY 20, 1993
# %    Gravity Wave Models for the Horizontal Wave Number Spectra
# %    of Atmospheric Velocity and Density Fluctuations
# %    CHESTER S. GARDNER, CHRIS A. HOSTETLER, AND STEVEN J. FRANKE
# %
# % The path from spectrum to velocity perturbation is followed using the 
# % approach described in Norris and Gibson (2002):
# %    INFRAMAP ENHANCEMENTS: ENVIRONMENTAL/PROPAGATION VARIABILITY AND
# %    LOCALIZATION ACCURACY OF INFRASONIC NETWORKS
# %    24th Seismic Research Review  Nuclear Explosion Monitoring: Innovation and Integration
# %    David E. Norris and Robert G. Gibson
# %
# %
# %  ==========================================
# %  NOTE ON THE WAVENUMBER m CONVENTION USED: 
# %  ==========================================
# %        Because this seems to be the convention used by Norris & Gibson,
# %        we assume the wavenumber m NOT to be the circular wave number,
# %        that is: it is equal to 1/lambda, and not 2*pi/lambda
# %        Actually, Figure 2 in Norris & Gibson should probably have the x-label
# %        "vertical wavenumber / (2*pi) (cycles/m)", and NOT "vertical wavenumber (cycles/m)"
# %        This would also be in line with Gardner 1993, Figure 1.

#################################
## Import event analyzer routines
import pandas as pd
from pdb import set_trace as bp
import numpy as np
import great_circle_calculator as gcc
from scipy import interpolate

def hann_onesided(x, winlen):

    w=1+0*np.array(x)
    
    g_ind = np.where (x <=winlen)
    N=np.shape(g_ind)[1]

    tmp = 1/2*(1-np.cos(np.pi*np.linspace(0,N-1,num=N)/N))
    w[g_ind]=tmp

    return w

def gaussian_shading(x,x_center,y,sigma):

    N=len(y)
    dx=x[2]-x[1]
    weights = np.exp(- (x - x_center)**2 / (2*sigma**2))

    y_shaded = y * weights

    return y_shaded, weights
        
def shade_and_sum_perts(perts,z_axis,z):

#% --- 0.8311*sigma is the 50% energy half-width of a Gaussian
        width_factor = 1 / 0.8311
        
#% --- The top altitude where we "cutoff" the perturbations        
        try:
            print('Z_max_cutoff=',z_max_cutoff)
        except:
            z_max_cutoff=110000
#% --- The bottom altitude where we "cutoff" the perturbations
        try:
            print('Z_min_falloffstart=',z_min_falloffstart)
        except:
            z_min_falloffstart=10000



        if z_max_cutoff <= np.max(z):
            print('max(z) cannot be greater than z_max_cutoff')
#% --- Memory allocation
        final_pert=np.zeros((np.shape(perts)))
        perts_shaded=np.zeros((np.shape(perts)))
        weights=np.zeros((np.shape(perts)))
        new_perts=np.zeros((np.shape(perts)))
#% --- Make sure the perts and z are in increasing order        
        sortinds=np.argsort(z)
        z=z[sortinds]
        sortinds=np.array(sortinds)
        for inx in range(0,np.shape(z)[0]):
                new_perts[:,inx]=perts[:,sortinds[inx]]
        perts=new_perts
#% --- Loop over all perts and shade them
        for ii in range(np.shape(perts)[1]):
            curr_pert = perts[:,ii]
            

            weights_lower = 0*np.array(z_axis)
            weights_upper = 0*np.array(z_axis)
             #% --- Shading of the "lower part"
            if ii==0:
                    sigma_lower = (z[0]/2) * width_factor
            else:
                    sigma_lower = (z[ii]-z[ii-1])/2 * width_factor
            g_ind_lower = np.where (z_axis < z[ii])
            # % --- Shading of the "upper part"
            perts_shaded[g_ind_lower,ii],weights_lower[g_ind_lower]=gaussian_shading(z_axis[g_ind_lower],z[ii],curr_pert[g_ind_lower],sigma_lower)
            if ii==len(z)-1:
                    sigma_upper = (z_max_cutoff - z[ii])/2 * width_factor
            else:
                    sigma_upper = (z[ii+1]-z[ii])/2 * width_factor
            g_ind_upper = np.where (z_axis >= z[ii])
            perts_shaded[g_ind_upper,ii],weights_upper[g_ind_upper]=gaussian_shading(z_axis[g_ind_upper],z[ii],curr_pert[g_ind_upper],sigma_upper)
            #bp()
        #% --- Accumulate final summed perturbation
            
        final_pert = np.sum(perts_shaded,axis=1)

#% --- Make sure to shade off appropriately towards ground level:
#   %     the perturbation must be zero at z_axis = 0
#   %     Use raised cosine shading
        w_lowercutoff=hann_onesided(z_axis, z_min_falloffstart)
        final_pert = final_pert * w_lowercutoff
        return final_pert, perts_shaded

class Gardner_realization():

    def __init__(self):
        self.params = {}

    def save_attributes(self, tot_pert, pert_components, z_axis, powspect, m, m_star_vec, mb_vec, z, stds_wanted):
        self.params['tot_pert'] = tot_pert
        self.params['pert_components'] = pert_components
        self.params['z_axis'] = z_axis
        self.params['powspect'] = powspect
        self.params['m'] = m
        self.params['m_star_vec'] = m_star_vec
        self.params['mb_vec'] = mb_vec
        self.params['z'] = z
        self.params['stds_wanted'] = stds_wanted

    def export(self, output_file):
        # export and save the perturbations for a realization - stdev couple
        np.savez(output_file, **self.params)

    def import_(self, input_file):
        npzfile = np.load(input_file)
        params = {}
        for file in npzfile.files:
            params[file] = npzfile[file]
        self.save_attributes(**params)

class Gardner_model():

    def __init__(self, options):
    
        # --- Prepare template indata struct
        self.z = options['z'];
        self.m_star_vec = options['m_star_vec'];
        self.mb_vec = options['mb_vec'];
        self.const = options['const'];
        self.m = options['m'];
        self.q = options['q'];
        self.s = options['s'];
        self.z_range_wanted = options['z_range_wanted'];
        self.stds_wanted_default = options['stds_wanted_default']
        self.amps = options['amps']
        self.N_realizations_per_set = options['N_realizations_per_set']
        self.output_dir = options['output_dir']
        
        #self.stds_wanted = stds_wanted_default .* AMPS(ll, :)

    def compute_Gardner_for_amps(self):
    
        """
        For a given list of amplitudes calculate the perturbations for a given number of realizations
        """
        
        dummy=0
 
        self.df_Perturbations = pd.DataFrame()
        for ii, amp in enumerate(self.amps):
        
            self.stds_wanted = self.stds_wanted_default*amp
            for jj in range(0,self.N_realizations_per_set):
                one_perturbation = self._get_gardner_realization();
                
                loc_df = pd.DataFrame()
                loc_df['z_axis'] = one_perturbation.params['z_axis']
                loc_df['wind']   = one_perturbation.params['tot_pert']
                for i_std, std in enumerate(self.stds_wanted):
                    loc_df['std-'+str(i_std)] = std
                loc_df['no-std'] = ii
                loc_df['no-per-std'] = jj
                
                self.df_Perturbations = self.df_Perturbations.append( loc_df )
                
        self.df_Perturbations.reset_index(drop=True, inplace=True)
        
    def _get_gardner_realization(self):
    
        stds_wanted = self.stds_wanted
        
        ## Calculate perturbations for each "level"            
        perts_initial, z_axis, stds_extracted_initial, \
            stds_fullperts_initial, powspect, amplspect = self._get_gardner_perts();
        
        ## Weight each perturbation level according to given "wanted standard deviations"
        w = stds_wanted / stds_fullperts_initial
        inx_a=np.shape(perts_initial)[0]
        inx_b=np.shape(perts_initial)[1]
        W=np.ones((inx_a,1))*np.transpose(w)

        perts = np.zeros((inx_a,inx_b))
        for inx in range(0,inx_a):
            for iny in range(0,inx_b): 
                perts[inx,iny]=perts_initial[inx,iny]*W[inx,iny]
                
        #perts=perts_initial*W
       
        stds = np.std(perts)
        
        ## Compound each perturbation component into a 1D vector "total perturbation"
        tot_pert, pert_components = shade_and_sum_perts(perts, z_axis, self.z)
        model = Gardner_realization()
        model.save_attributes(tot_pert, pert_components, z_axis, powspect, \
            self.m, self.m_star_vec, self.mb_vec, self.z, self.stds_wanted)
            
        return model

    # z_axis is the altitude axis
    # z is the "center" of each perturbation realization
    # perts has each realization as in a column vector
    def _get_gardner_spect(self):
#
        """
        NOTE ON THE INPUT:
             m_star_vec, mb_vec, m here must be the wavenumber as 1/lambda, and NOT
             the wavenmuber as defined by 2*pi/lamba (the circular wavenumber)
        
        OUTPUT:
        Power spectral density from Gardner model [m/s]^2 * [m]^(-1)
        """

        N_mstar = len(self.m_star_vec);

        if not N_mstar == len(self.mb_vec):
                sys.exit('length(m_star_vec) must be equal to length(mb_vec)');
        
        pow_spect = np.zeros((len(self.m), len(self.m_star_vec)));

        # --- Loop over the scale heights
        for ii in range(0, N_mstar):

            # --- Unsaturated gravity wave region
            g_ind = np.where(self.m <= self.m_star_vec[ii])[0] 
            pow_spect[g_ind, ii] = self.const * self.m[g_ind]**self.s / self.m_star_vec[ii]**(3+self.s);

            # --- Saturated gravity wave region
            g_ind = np.where((self.m > self.m_star_vec[ii]) & (self.m <= self.mb_vec[ii]))[0]
            pow_spect[g_ind, ii] = self.const * self.m[g_ind]**(-self.q);

            # --- Kolmogorov power-law tail
            g_ind = np.where(self.m > self.mb_vec[ii])[0]
            pow_spect[g_ind, ii] = self.const * self.m[g_ind]**(-5/3) * self.mb_vec[ii]**(5/3 -3);

        ampl_spect = np.sqrt(pow_spect);

        return pow_spect, ampl_spect 

    def _get_gardner_perts(self):

        ## Calculate spectrum for each m*
        powspect, amplspect = self._get_gardner_spect();

        ## Add random phase and invert transform to get real
        ## perturbations for each phase
        [perts, z_axis, stds_fullperts] = self._get_perts_from_spect(self.m, amplspect);
        
        ## Possibly output only a selected z altitude range
        ## perts generated have different amplitudes in matlab and python after first weighting.
        if np.shape(self.z_range_wanted)[0]>0:
        
            if self.z_range_wanted[1] <= self.z_range_wanted[0]:
                error('z_range_wanted must be increasing')
            

            g_ind = np.where( (z_axis >= self.z_range_wanted[0]) & (z_axis <= self.z_range_wanted[1]));
            z_axis = z_axis[g_ind];
            perts  = perts[g_ind, :];
            perts  = np.squeeze(perts)   

        stds_extracted = np.std(perts);

        return perts, z_axis, stds_extracted, stds_fullperts, powspect, amplspect

    def _get_perts_from_spect(self, m, ampl_spect):

        M, N = ampl_spect.shape

        ## total number of samples in freq. domain
        M_full = (M+M-1); 

        ## Get the altitude axis in the spatial domain
        dm = m[1]-m[0]
        dz = 1 / (M_full * dm) 
        z_axis = dz * np.array(np.linspace(0,M_full-1,num=M_full))

        ## [m], provided that m is in 1/meters
        z_axis = z_axis - np.mean(z_axis);

        ## Allocate memory
        #perts = np.array(0*[ampl_spect, ampl_spect[2:,:]])
        perts=np.zeros(((len(ampl_spect)-1)*2,N))
        
        # --- Set random generator seed
        #rng(1, 'twister');
        #rand('state', 0);

        # --- Main loop
        for ii in range(N):
        
            randphase = 2*np.pi*np.random.rand(M, 1);
            #randphase = 2*np.pi*np.ones((M, 1));
            toti=np.exp(1j*randphase)
            tmp_posfreq = np.array(ampl_spect[:, ii]).reshape(len(toti),1).tolist() * toti.reshape(len(toti),1);

            # --- Note that spect is the power spectrum, while we want
            #     the amplitude spectrum.
            #     Therefore take square root:
            #   tmp_posfreq = sqrt(spect(:, ii)) .* exp(i*randphase);
            tmp_negfreq = np.flipud(np.conj(tmp_posfreq[2:])); # Symmetry in order to get real ifft result
            ampspect_allfreqs = np.append(tmp_posfreq, tmp_negfreq); 
            s=np.fft.irfft(ampspect_allfreqs,len(ampspect_allfreqs))

            perts[:, ii] = np.sqrt(M_full) * np.fft.fftshift(s);  

	# sqrt(M_full) factor: to make MATLAB's ifft transform unitary
        #disp(['std(perts) = ' mat2str(std(perts))]);
        stds=np.zeros((np.shape(perts)[1]))
        for inx in range(0, np.shape(perts)[1]):
            stds[inx]=np.std(perts[:,inx])

        return perts, z_axis, stds

    def save_model(self, ext=''):
        
        file = '{output_dir}dataset{ext}.csv'.format(output_dir=self.output_dir, ext=ext)
        self.df_Perturbations.to_csv(file, sep=',', index=False, header=True)

def get_one_realization(stds, range_altitudes=[0, 150], altitude_levels=[ 84, 70, 45, 21], N_realizations_per_set=1):

    """
    Compute one Gardner realization for a list of stds
    """

    options={}
    # --- Alitude levels for each perturbation component
    options['z'] = 1000*np.array(altitude_levels) # m Norris & Gibson 2002
    options['m_star_vec'] = [8.8619*0.00001, 1.3226*0.0001, 2.7120*0.0001, 5.3350*0.0001] # Norris & Gibson 2002
    options['mb_vec'] = [1.7017*0.001, 2.5398*0.001, 5.2076*0.001, 1.0104*0.01] # Brunt-V freqs Norris & Gibson 2002
    options['const'] = 1
    m_1 = 0;
    m_2 = 0.01; 
    N_m = 20001;
    options['m'] = np.linspace(m_1, m_2, N_m).T
    options['q'] = 3
    options['s'] = 2
    
    # --- Profile alitutde limits
    options['z_range_wanted'] = 1000* np.array(range_altitudes) # [m]
    # --- The standard deviation wanted for each of the "perturbation levels"
    options['stds_wanted_default'] = stds
    default = np.array([1, 1, 1, 1])
    options['amps'] = [ default ]
    options['N_realizations_per_set'] = N_realizations_per_set
    
    model = Gardner_model(options)
    model.compute_Gardner_for_amps()
    
    return model.df_Perturbations

def add_perturbations_to_profile(profile, stds, point_start, point_end, range_altitudes=[0, 150], 
                                 altitude_levels=[ 84, 70, 45, 21], N_realizations_per_set=1, seed_Gardner=-1):

    """
    Add Gardner perturbation to wind profiles for a given std list at various altitudes
    Altitude in profile should be in km
    """
    
    if seed_Gardner > -1:
        np.random.seed(seed_Gardner)
    
    az = gcc.bearing_at_p2(point_start[::-1], point_end[::-1])
    
    df_perturbations = get_one_realization(stds, range_altitudes=range_altitudes, 
                            altitude_levels=altitude_levels, N_realizations_per_set=N_realizations_per_set)
    z = df_perturbations.z_axis.values/1e3
    z[0] = 0.
    perturbations = df_perturbations.wind.values
    f = interpolate.interp1d(z, perturbations, kind='cubic')
    perturbations = f(profile['z'].values)
    
    cos_az = abs(np.cos(np.radians(az)))
    profile['v'] += cos_az * perturbations
    profile['u'] += (1.-cos_az) * perturbations
    
######################################################
if __name__ == '__main__':

    options={}
    
    ## Alitude levels for each perturbation component
    options['z'] = 1000*np.array([ 84, 70, 45, 21]) # m Norris & Gibson 2002
    options['m_star_vec'] = [8.8619*0.00001, 1.3226*0.0001, 2.7120*0.0001, 5.3350*0.0001] # Norris & Gibson 2002
    options['mb_vec'] = [1.7017*0.001, 2.5398*0.001, 5.2076*0.001, 1.0104*0.01] # Brunt-V freqs Norris & Gibson 2002
    options['const'] = 1
    m_1 = 0;
    m_2 = 0.01; 
    N_m = 20001;
    options['m'] = np.linspace(m_1, m_2, N_m).T
    options['q'] = 3
    options['s'] = 2
    
    ## Profile alitutde limits
    options['z_range_wanted'] = 1000* np.array([0, 150]) # [m]
    
    ## The standard deviation wanted for each of the "perturbation levels"
    options['stds_wanted_default'] = [22.53, 15.13, 7.4, 3.73]
    default = np.array([1, 1, 1, 1])
    options['amps'] = [ default, 1.3 * default, 1.6 * default, 2   * default, 1.1 * default,
      1.15* default, 
      1.2 * default, 
      1.25* default, 
      0.80* default,
      0.60* default]
    options['N_realizations_per_set'] = 1
    options['output_dir']='/staff/antoine/Projects/Kiruna/Gardner_model/'
    
    ## Compute Gardner perturbations
    model = Gardner_model(options)
    model.compute_Gardner_for_amps()
    model.save_model()
