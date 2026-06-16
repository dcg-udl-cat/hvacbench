from enum import StrEnum


class EnergyPriceType(StrEnum):
    CONSTANT = "constant"  # Completely constant electricity price
    DYNAMIC = "dynamic" # Electricity price for a day / night tariff
    HIGHLY_DYNAMIC = "highly_dynamic" # Spot electricity price
