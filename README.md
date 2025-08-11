# OD-filter-testing
Filter comparison program, explicitly for near rectilinear halo orbit (NRHO) determination, at its current stage. Further development would see this integrated with an orbit propagator, such that other orbit types could be easily selected and tested.

The program is designed to be run from a main interface script included below. This interface allows the user to upload a truth trajectory, a corresponding time dataset, and convert the truth to a reference trajectory via measurement noise which is also set in this script. Process noise is also set here, as well as 'consider' parameters (sources of uncertainty) and their covariance for use in the unscented Schmidt Kalman filter (USKF). 

The interface allows you to set which filter to run out of the following filters:
- Extended Kalman Filter (EKF)
- Cubature Kalman Filter (CKF)
- Unscented Kalman Filter (UKF) 

An additional hybridisation option is also included which allows any of the aforementioned filters to be implemented together. This hybrid filter option was implemented specifically with NRHO OD in mind, as the main issue with NRHOs is that towards the perilune their dynamics become increasingly unstable, and thus far more difficult for simpler and more efficient filters to achieve adequate OD results. Thus, testing an option that allows for implementation of more complex filters only when they are needed may provide a computational advantage. Currently, for the hybridisation option, boundary conditions are user set manually, with the hybrid function then sorting the trajectory data into stable and unstable regions for computation. This will allow testing on the impacts of broadening and narrowing the unstable region in which the more advanced filter runs on the OD accuracy. 

This interface lastly generates visualisation and error plots for direct comparison between the truth trajectory and filter results.

Note that the HALO propagator has now been integrated into the program, and that propagation no longer relies on CR3BP modelling. 

This project is largely developed using the HALO propagator, link to GitHub repositary here: [High-precision-Analyser-of-Lunar-Orbits](https://github.com/yang-researchgroup/High-precision-Analyser-of-Lunar-Orbits). 

There are additional files required to be installed before running this tool. Firstly, run:
pip install -r requirements.txt
This will install the dependencies listed in the text file. 

Secondly, de430.bsp must be downloaded  (download here: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/) and installed into multiple locations:
- the main folder 'main'
- input
- ker
- visualisation processes
- visualisation processes --> input
