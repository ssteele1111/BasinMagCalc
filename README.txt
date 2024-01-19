Reproduction code for: Could the weak magnetism of Martian impact basins reflect cooling in a reversing dynamo?

System requirements:
	Tested on Windows 10 22H2 and Rocky Linux 8.7 (Green Obsidian)

	Python 3.8.3-3.10.9
	Required Python packages:
	numpy,pandas,matplotlib,time,os,numba,meshio,asyncio,joblib


Installation:
	Create a new Conda/Mamba/etc. environment and install the required Python packages. 
	
	Estimated install time: 20 minutes


Demo instructions:
	Running basiniter.py as-is from within the BasinMagCalc folder will perform the magnetic field calculations 
	described in the manuscript Methods section over an 80x2500km region at 200 km altitude for a 600 km demo basin. 
	The included demo will test 5 random reversal histories and save the output in \600km\mag_output\. 
	
	Running this code will output:
	1. BMaps: Final magnetic field maps
	2. BMaps_nolr: Magnetic field maps neglecting late remagnetization
	3. RevRates: Mean reversal rates
	4. Revs: Full reversal histories
	5. SuscMaps: Subsampled maps of magnetization within the basin (note that this is subsampled aggressively
		to storage limitations, so cannot be used to regenerate magnetic fields later.)
	
	Expected output is included in \600km\mag_output_expected\. 
	
	Estimated run time: 15 minutes
	
	Also included are some simple analysis tools in mag_data_analysis.nb. Note that the magnetization 
	intensity assumed in the magnetic field calculation scripts is 1 A/m; since magnetic field strength scales
	linearly with magnetization intensity, one can multiply final magnetic field maps by the desired amount
	to compare to manuscript values.
	
	
Further use/reproduction instructions:
	Any or all data in the manuscript can be reproduced by downloading thermal histories for different basin
	sizes from the Harvard Dataverse:
		https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/TPW4WB&faces-redirect=true
		
	and correspondingly changing input/output directories in basiniter.py. Different reversal frequencies can 
	be tested by changing the mu and thresh parameters in the do_basiniter_dual() function call. The field 
	mapping domain can be changed by defining a new imgrid or choosing one of the different supplied imgrids.

	Note: performing these calculations for larger basins, or for larger imgrids, can take many hours and be 
	VERY resource intensive. Large-domain mapping for the largest basins typically uses ~200 GB memory! Be 
	cautious attempting calculations for any basin sizes larger than 600-800km diameter on a normal desktop. 
	