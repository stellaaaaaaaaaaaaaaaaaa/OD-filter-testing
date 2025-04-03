# OD-filter-testing - filter algorithms
This section includes the scripts and corresponding equations for these filter algorithms. 

As previously mentioned the filter algorithms included are as follows:
- Extended Kalman Filter (EKF)
- Cubature Kalman Filter (CKF)
- Square Root Cubature Kalman Filter (SRCKF)
- Unscented Schmidt Kalman Filter (USKF)
- Unscented Kalman Filter (UKF)

EKF and UKF were selected due to their particular prevalence in OD applications, and as such have long been a focus in NRHO OD research. 
CKF was selected due to its similarity to UKF, with a potential improvement upon computational cost as the points propagated follow the 'cubature rule' over the unscented transform (UT).
Furthermore SRCKF was selected due to its potentially improved stability, as working with square root matrices ensures the results are always positive definite as required. 
USKF was the final filter selected, as once again it improves upon the UKF and the other filters by incorporating consider parameters to ensure that the environmental uncertainties are accounted for in the OD result. 

These reasons will be further expanded upon in the final thesis paper release. 
