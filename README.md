Repo containing common inversion routines for Ly-alpha data on Alcator C-Mod and Balmer-alpha data on MAST-U.


# Tomas_inversion.py

This is the inversion routine that Tomas sent Andres recently (with backprojection fixed and option to change the regularisation). I tidied the GUI a little bit and also added showing the GUI as an optional keyword argument.

# Jamie_inversion.py

This is a version of the inversion that I've been using for C-Mod. It contains a number of extra keyword arguments (some specific to C-Mod):

- calib_factor: a calibration factor that can be applied to the raw data
- apply offsets: some ad-hoc offset calculations (e.g WB4LY camera sometimes has a dead-channel which only picks up dark current and can therefore be subtracted from other channels to reduce noise
- t_window: if only a certain part of the shot needs to be inverted
- smooth_brightness_in_time: option to use a savitzky-golay filter to smooth the raw brightness signals in time before inverting.

# TODO:
- [ ] Add in regularisation as a key-word argument
- [ ] Check backprojections for the shot that Jamie send Steven
- [ ] Work out what dL is doing in the inversions
- [ ] Work out how to invert for non-horizontal views (does it even make a difference??)
- [ ] Merge Jamie's inversion with Tomas' inversion (probably doesn't need to be two separate files).


