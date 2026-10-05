"""
units.py — Civil Engineering Unit Conversion System

Provides:
  - UNIT_CONVERSIONS: complete conversion table for all civil engineering units
  - convert(): convert a value between any two units in the same category
  - get_available_units(): units grouped by category
  - format_unit_value(): format a value with unit for display
"""

from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# Complete unit conversion table
# Structure: { category: { unit_symbol: factor_to_SI_base } }
# To convert from unit_A to unit_B:
#   value_SI = value_A * factor_A
#   value_B  = value_SI / factor_B
# ---------------------------------------------------------------------------
UNIT_CONVERSIONS: dict[str, dict[str, float]] = {

    # -----------------------------------------------------------------------
    # Length
    # SI base: metre (m)
    # -----------------------------------------------------------------------
    "Panjang (Length)": {
        "mm":   1e-3,         # millimetre
        "cm":   1e-2,         # centimetre
        "dm":   0.1,          # decimetre
        "m":    1.0,          # metre (SI base)
        "km":   1e3,          # kilometre
        "in":   0.0254,       # inch
        "ft":   0.3048,       # foot
        "yd":   0.9144,       # yard
        "mi":   1609.344,     # mile (statute)
        "nmi":  1852.0,       # nautical mile
        "ch":   20.1168,      # chain (surveying)
        "fur":  201.168,      # furlong
        "fath": 1.8288,       # fathom
    },

    # -----------------------------------------------------------------------
    # Area
    # SI base: square metre (m²)
    # -----------------------------------------------------------------------
    "Luas (Area)": {
        "mm²":    1e-6,
        "cm²":    1e-4,
        "dm²":    0.01,
        "m²":     1.0,
        "km²":    1e6,
        "ha":     1e4,        # hectare
        "a":      100.0,      # are
        "in²":    6.4516e-4,
        "ft²":    0.09290304,
        "yd²":    0.83612736,
        "mi²":    2.589988e6,
        "ac":     4046.856,   # acre
    },

    # -----------------------------------------------------------------------
    # Volume
    # SI base: cubic metre (m³)
    # -----------------------------------------------------------------------
    "Volume": {
        "mm³":    1e-9,
        "cm³":    1e-6,
        "mL":     1e-6,       # millilitre = cm³
        "L":      1e-3,       # litre
        "m³":     1.0,
        "km³":    1e9,
        "in³":    1.6387e-5,
        "ft³":    0.028317,
        "yd³":    0.764555,
        "gal_US": 3.78541e-3, # US gallon
        "gal_UK": 4.54609e-3, # Imperial gallon
        "bbl":    0.158987,   # oil barrel
        "qt_US":  9.46353e-4, # US quart
    },

    # -----------------------------------------------------------------------
    # Mass
    # SI base: kilogram (kg)
    # -----------------------------------------------------------------------
    "Massa (Mass)": {
        "mg":     1e-6,       # milligram
        "g":      1e-3,       # gram
        "kg":     1.0,        # kilogram (SI base)
        "t":      1e3,        # tonne (metric ton)
        "oz":     0.028350,   # ounce (avoirdupois)
        "lb":     0.453592,   # pound (avoirdupois)
        "ton_UK": 1016.047,   # long ton
        "ton_US": 907.1847,   # short ton
        "gr":     6.4799e-5,  # grain
        "cwt_UK": 50.8023,    # hundredweight (UK)
    },

    # -----------------------------------------------------------------------
    # Force
    # SI base: Newton (N)
    # -----------------------------------------------------------------------
    "Gaya (Force)": {
        "N":    1.0,
        "kN":   1e3,
        "MN":   1e6,
        "kgf":  9.80665,     # kilogram-force
        "tf":   9806.65,     # tonne-force
        "lbf":  4.44822,     # pound-force
        "kip":  4448.22,     # kilopound-force
        "dyn":  1e-5,        # dyne (CGS)
        "pdl":  0.138255,    # poundal
    },

    # -----------------------------------------------------------------------
    # Pressure / Stress
    # SI base: Pascal (Pa = N/m²)
    # -----------------------------------------------------------------------
    "Tekanan/Tegangan (Pressure/Stress)": {
        "Pa":    1.0,
        "kPa":   1e3,
        "MPa":   1e6,
        "GPa":   1e9,
        "bar":   1e5,
        "mbar":  100.0,
        "atm":   101325.0,
        "psi":   6894.757,   # pound per square inch
        "psf":   47.8803,    # pound per square foot
        "ksi":   6.894757e6, # kilopound per square inch
        "kgf/cm²": 98066.5, # kilogram-force per cm²
        "tf/m²": 9806.65,   # tonne-force per m²
        "kN/m²": 1e3,       # = kPa
        "MN/m²": 1e6,       # = MPa
        "mm Hg":  133.322,  # millimetre of mercury
        "m H₂O":  9806.65,  # metre of water column
    },

    # -----------------------------------------------------------------------
    # Density
    # SI base: kg/m³
    # -----------------------------------------------------------------------
    "Densitas (Density)": {
        "kg/m³":   1.0,
        "g/cm³":   1000.0,
        "g/mL":    1000.0,
        "t/m³":    1000.0,
        "kg/L":    1000.0,
        "lb/ft³":  16.01846,
        "lb/gal":  119.8264,
        "lb/in³":  27679.9,
        "kN/m³":   101.9716,  # unit weight conversion
    },

    # -----------------------------------------------------------------------
    # Unit Weight (Specific Weight = Weight per Volume)
    # SI base: N/m³
    # -----------------------------------------------------------------------
    "Berat Satuan (Unit Weight)": {
        "N/m³":   1.0,
        "kN/m³":  1e3,
        "kgf/m³": 9.80665,
        "tf/m³":  9806.65,
        "lb/ft³": 157.0875,
        "lbf/ft³":157.0875,
        "pcf":    157.0875,   # pound-force per cubic foot
    },

    # -----------------------------------------------------------------------
    # Velocity / Speed
    # SI base: m/s
    # -----------------------------------------------------------------------
    "Kecepatan (Velocity)": {
        "m/s":    1.0,
        "km/h":   1.0 / 3.6,
        "cm/s":   0.01,
        "mm/s":   0.001,
        "ft/s":   0.3048,
        "mph":    0.44704,
        "knot":   0.514444,
        "m/min":  1.0 / 60.0,
        "m/hr":   1.0 / 3600.0,
    },

    # -----------------------------------------------------------------------
    # Discharge (Flow Rate)
    # SI base: m³/s
    # -----------------------------------------------------------------------
    "Debit (Discharge)": {
        "m³/s":     1.0,
        "m³/hr":    1.0 / 3600.0,
        "m³/day":   1.0 / 86400.0,
        "L/s":      1e-3,
        "L/min":    1e-3 / 60.0,
        "L/hr":     1e-3 / 3600.0,
        "mL/s":     1e-6,
        "ft³/s":    0.028317,   # cfs
        "ft³/min":  4.71947e-4,
        "gal_US/min": 6.30902e-5,  # gpm
        "gal_US/hr":  1.05150e-6,
        "gal_UK/min": 7.57682e-5,
        "Mm³/day":  11.5741,    # million m³/day
    },

    # -----------------------------------------------------------------------
    # Angle
    # SI base: radian (rad)
    # -----------------------------------------------------------------------
    "Sudut (Angle)": {
        "rad":    1.0,
        "deg":    0.017453293,  # π/180
        "grad":   0.015707963,  # π/200
        "mrad":   0.001,
        "arcmin": 2.908882e-4,  # degree/60
        "arcsec": 4.848137e-6,  # degree/3600
        "turn":   6.283185307,  # full circle
    },

    # -----------------------------------------------------------------------
    # Temperature  ← NOTE: affine transformation, not pure factor
    # Stored separately as (scale, offset_to_celsius)
    # -----------------------------------------------------------------------
    # Temperature handled specially in convert() function

    # -----------------------------------------------------------------------
    # Time
    # SI base: second (s)
    # -----------------------------------------------------------------------
    "Waktu (Time)": {
        "s":      1.0,
        "min":    60.0,
        "hr":     3600.0,
        "day":    86400.0,
        "week":   604800.0,
        "month":  2.628e6,      # 30.44 days average
        "year":   3.156e7,      # 365.25 days
        "ms":     0.001,
        "μs":     1e-6,
    },

    # -----------------------------------------------------------------------
    # Energy
    # SI base: Joule (J)
    # -----------------------------------------------------------------------
    "Energi (Energy)": {
        "J":      1.0,
        "kJ":     1e3,
        "MJ":     1e6,
        "GJ":     1e9,
        "cal":    4.18680,
        "kcal":   4186.80,
        "Wh":     3600.0,
        "kWh":    3.6e6,
        "MWh":    3.6e9,
        "BTU":    1055.056,
        "ft·lbf": 1.35582,
        "eV":     1.60218e-19,
        "erg":    1e-7,
    },

    # -----------------------------------------------------------------------
    # Power
    # SI base: Watt (W)
    # -----------------------------------------------------------------------
    "Daya (Power)": {
        "W":      1.0,
        "kW":     1e3,
        "MW":     1e6,
        "GW":     1e9,
        "HP":     745.6999,    # horsepower (mechanical)
        "PS":     735.49875,   # horsepower (metric, Pferdestärke)
        "BTU/hr": 0.29307,
        "kcal/hr":1.16300,
        "ft·lbf/s":1.35582,
    },

    # -----------------------------------------------------------------------
    # Permeability (Hydraulic Conductivity)
    # SI base: m/s
    # -----------------------------------------------------------------------
    "Permeabilitas (Hydraulic Conductivity)": {
        "m/s":   1.0,
        "cm/s":  0.01,
        "mm/s":  0.001,
        "m/hr":  1.0 / 3600.0,
        "m/day": 1.0 / 86400.0,
        "ft/s":  0.3048,
        "ft/day":3.528e-6,
        "darcy": 9.869e-13,    # 1 darcy (petroleum engineering)
        "mD":    9.869e-16,    # millidarcy
    },

    # -----------------------------------------------------------------------
    # Consolidation Coefficient Cv
    # SI base: m²/s
    # -----------------------------------------------------------------------
    "Koefisien Konsolidasi Cv": {
        "m²/s":    1.0,
        "m²/yr":   3.171e-8,
        "cm²/s":   1e-4,
        "cm²/min": 1e-4 / 60.0,
        "ft²/yr":  9.29e-2 / 3.156e7,
        "mm²/yr":  1e-6 / 3.156e7,
    },

    # -----------------------------------------------------------------------
    # Rainfall Intensity
    # SI base: mm/hr
    # -----------------------------------------------------------------------
    "Intensitas Hujan (Rainfall Intensity)": {
        "mm/hr":   1.0,
        "mm/min":  60.0,
        "mm/s":    3600.0,
        "cm/hr":   10.0,
        "m/hr":    1000.0,
        "in/hr":   25.4,
        "in/day":  25.4 / 24.0,
    },
}

# ---------------------------------------------------------------------------
# Temperature — special handling (affine transforms)
# Structure: { symbol: (scale_factor_to_celsius, offset_celsius) }
# T_celsius = T_unit * scale + offset
# T_unit = (T_celsius - offset) / scale
# ---------------------------------------------------------------------------
TEMPERATURE_UNITS: dict[str, tuple[float, float]] = {
    "°C": (1.0,       0.0),        # Celsius (base)
    "°F": (5.0/9.0, -32.0*5.0/9.0),  # Fahrenheit → (F-32)*5/9 → C
    "K":  (1.0,     -273.15),      # Kelvin → K - 273.15 → C
    "°R": (5.0/9.0, -491.67*5.0/9.0), # Rankine
}


# ---------------------------------------------------------------------------
# Helper: build a flat list of all units
# ---------------------------------------------------------------------------
def get_available_units(category: Optional[str] = None) -> dict[str, list[str]]:
    """
    Return available units grouped by category.

    Parameters
    ----------
    category : str, optional
        If provided, return only units for that category.
        Use None to return all categories.

    Returns
    -------
    dict[str, list[str]]
        Mapping from category name → list of unit symbols.

    Examples
    --------
    >>> get_available_units("Panjang (Length)")
    {'Panjang (Length)': ['mm', 'cm', 'dm', 'm', 'km', ...]}

    >>> get_available_units()
    {'Panjang (Length)': [...], 'Luas (Area)': [...], ...}
    """
    result: dict[str, list[str]] = {}

    if category is not None:
        if category in UNIT_CONVERSIONS:
            result[category] = list(UNIT_CONVERSIONS[category].keys())
        elif category == "Suhu (Temperature)":
            result[category] = list(TEMPERATURE_UNITS.keys())
        else:
            raise ValueError(
                f"Category '{category}' not found. "
                f"Available: {list(UNIT_CONVERSIONS.keys()) + ['Suhu (Temperature)']}"
            )
        return result

    # All categories
    for cat, units in UNIT_CONVERSIONS.items():
        result[cat] = list(units.keys())
    result["Suhu (Temperature)"] = list(TEMPERATURE_UNITS.keys())

    return result


# ---------------------------------------------------------------------------
# Core conversion function
# ---------------------------------------------------------------------------

def convert(
    value: float,
    from_unit: str,
    to_unit: str,
) -> float:
    """
    Convert a value from one unit to another.

    Supports all units in UNIT_CONVERSIONS and TEMPERATURE_UNITS.
    Both units must belong to the same physical quantity (category).

    Parameters
    ----------
    value : float
        Numeric value in ``from_unit``.
    from_unit : str
        Source unit symbol (e.g. "kPa", "ft", "L/s").
    to_unit : str
        Target unit symbol.

    Returns
    -------
    float
        Converted value in ``to_unit``.

    Raises
    ------
    ValueError
        If either unit is not found, or if they belong to different categories.

    Examples
    --------
    >>> convert(1.0, "kPa", "psi")
    0.14504

    >>> convert(25.0, "°C", "°F")
    77.0

    >>> convert(1.0, "m³/s", "L/s")
    1000.0
    """
    if from_unit == to_unit:
        return value

    # -----------------------------------------------------------------------
    # Temperature: affine conversion
    # -----------------------------------------------------------------------
    if from_unit in TEMPERATURE_UNITS and to_unit in TEMPERATURE_UNITS:
        scale_from, offset_from = TEMPERATURE_UNITS[from_unit]
        scale_to, offset_to = TEMPERATURE_UNITS[to_unit]
        # Convert from_unit → Celsius
        celsius = value * scale_from + offset_from
        # Convert Celsius → to_unit: T_to = (celsius - offset_to) / scale_to
        result = (celsius - offset_to) / scale_to
        return round(result, 6)

    if from_unit in TEMPERATURE_UNITS or to_unit in TEMPERATURE_UNITS:
        raise ValueError(
            f"Cannot mix temperature unit '{from_unit}' or '{to_unit}' "
            "with a non-temperature unit."
        )

    # -----------------------------------------------------------------------
    # Find categories for from_unit and to_unit
    # -----------------------------------------------------------------------
    from_category: Optional[str] = None
    to_category: Optional[str] = None

    for cat, units in UNIT_CONVERSIONS.items():
        if from_unit in units:
            from_category = cat
        if to_unit in units:
            to_category = cat

    if from_category is None:
        raise ValueError(
            f"Unit '{from_unit}' not found in conversion table. "
            "Use get_available_units() to see available units."
        )
    if to_category is None:
        raise ValueError(
            f"Unit '{to_unit}' not found in conversion table. "
            "Use get_available_units() to see available units."
        )
    if from_category != to_category:
        raise ValueError(
            f"Cannot convert between different physical quantities: "
            f"'{from_unit}' is in '{from_category}', "
            f"'{to_unit}' is in '{to_category}'."
        )

    factor_from = UNIT_CONVERSIONS[from_category][from_unit]
    factor_to = UNIT_CONVERSIONS[from_category][to_unit]

    # Convert: value → SI base → to_unit
    value_si = value * factor_from
    result = value_si / factor_to

    return result


# ---------------------------------------------------------------------------
# Reverse lookup: find all units for a given symbol
# ---------------------------------------------------------------------------

def find_unit_category(unit: str) -> Optional[str]:
    """
    Find the category name for a given unit symbol.

    Parameters
    ----------
    unit : str
        Unit symbol to look up.

    Returns
    -------
    str or None
        Category name, or None if not found.
    """
    if unit in TEMPERATURE_UNITS:
        return "Suhu (Temperature)"
    for cat, units in UNIT_CONVERSIONS.items():
        if unit in units:
            return cat
    return None


# ---------------------------------------------------------------------------
# Display formatting
# ---------------------------------------------------------------------------

def format_unit_value(
    value: float,
    unit: str,
    decimals: int = 3,
    use_scientific: bool = False,
    thousands_sep: bool = False,
) -> str:
    """
    Format a numeric value with its unit for human-readable display.

    Parameters
    ----------
    value : float
        The numeric value.
    unit : str
        Unit symbol string.
    decimals : int, optional
        Number of decimal places, default 3.
    use_scientific : bool, optional
        Use scientific notation for very large/small values, default False.
    thousands_sep : bool, optional
        Add thousands separator for large numbers, default False.

    Returns
    -------
    str
        Formatted string, e.g. "123.456 kPa" or "1.23 × 10⁻³ m/s".

    Examples
    --------
    >>> format_unit_value(123456.789, "kPa", decimals=2)
    '123456.79 kPa'

    >>> format_unit_value(0.000123, "m/s", use_scientific=True)
    '1.230 × 10⁻⁴ m/s'
    """
    if use_scientific or (abs(value) < 0.001 and value != 0.0) or abs(value) >= 1e7:
        # Format as scientific notation
        fmt = f"{value:.{decimals}e}"
        # Replace Python scientific notation with nicer Unicode
        if "e" in fmt:
            mantissa, exp_str = fmt.split("e")
            exp_int = int(exp_str)
            superscript_map = {
                "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
                "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
                "-": "⁻", "+": "",
            }
            exp_display = "".join(superscript_map.get(c, c) for c in str(exp_int))
            formatted = f"{mantissa} × 10{exp_display}"
        else:
            formatted = fmt
    else:
        if thousands_sep:
            formatted = f"{value:,.{decimals}f}"
        else:
            formatted = f"{value:.{decimals}f}"

    return f"{formatted} {unit}"


# ---------------------------------------------------------------------------
# Batch conversion helper
# ---------------------------------------------------------------------------

def convert_dataframe_column(
    df: "pd.DataFrame",  # type: ignore  # noqa: F821
    column: str,
    from_unit: str,
    to_unit: str,
    new_column: Optional[str] = None,
) -> "pd.DataFrame":  # type: ignore  # noqa: F821
    """
    Apply unit conversion to an entire DataFrame column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column to convert.
    from_unit : str
        Source unit.
    to_unit : str
        Target unit.
    new_column : str, optional
        Name for the result column. If None, overwrites the original column.

    Returns
    -------
    pd.DataFrame
        DataFrame with converted values (copy, does not modify original).
    """
    import pandas as pd  # lazy import

    result = df.copy()
    target_col = new_column if new_column else column
    result[target_col] = df[column].apply(
        lambda x: convert(float(x), from_unit, to_unit) if pd.notna(x) else x
    )
    return result


# ---------------------------------------------------------------------------
# Convenience: common civil engineering conversions
# ---------------------------------------------------------------------------

def kpa_to_kgfcm2(value_kpa: float) -> float:
    """Convert kPa to kgf/cm²."""
    return convert(value_kpa, "kPa", "kgf/cm²")


def kgfcm2_to_kpa(value: float) -> float:
    """Convert kgf/cm² to kPa."""
    return convert(value, "kgf/cm²", "kPa")


def gcm3_to_knm3(value_gcm3: float) -> float:
    """Convert g/cm³ to kN/m³ (unit weight)."""
    # γ [kN/m³] = ρ [g/cm³] × 9.81
    return value_gcm3 * 9.81


def knm3_to_gcm3(value_knm3: float) -> float:
    """Convert kN/m³ to g/cm³."""
    return value_knm3 / 9.81