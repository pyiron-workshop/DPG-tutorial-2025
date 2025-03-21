from pyiron_workflow import as_function_node, as_macro_node, as_dataclass_node
import numpy as np

@as_function_node
def ConvertLoadToStress(df, area: float):
    """
    Read in csv file, convert load to stress
    """
    kN_to_N = 0.001  # convert kiloNewton to Newton
    mm2_to_m2 = 1e-6  # convert square millimeters to square meters
    df["Stress"] = df["Load"] * kN_to_N / (area * mm2_to_m2)
    #although it says extensometer elongation, the values are in percent! 
    strain = df["Extensometer elongation"].values.flatten()
    #subtract the offset from the dataset
    strain = strain - strain[0]
    stress = df["Stress"].values.flatten()
    return stress, strain

@as_function_node
def CalculateYoungsModulus(stress: np.ndarray, strain: np.ndarray, strain_cutoff: float =0.2) -> float:
    """
    Calculate the Youngs modulus with a linear fit
    """
    import numpy as np
    percent_to_fraction = 100  # convert
    MPa_to_GPa = 1 / 1000  # convert MPa to GPa
    arg = np.argsort(np.abs(np.array(strain) - strain_cutoff))[0]
    fit = np.polyfit(strain[:arg], stress[:arg], 1)
    youngs_modulus = fit[0] * percent_to_fraction * MPa_to_GPa
    return youngs_modulus
