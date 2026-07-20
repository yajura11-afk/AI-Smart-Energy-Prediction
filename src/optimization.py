"""Energy consumption optimization and recommendation utilities."""


ELECTRICITY_COST_PER_KWH = 8


def optimize_energy(predicted_consumption: float) -> dict:
    """Generate recommendations and estimate potential savings."""

    if predicted_consumption >= 3:
        category = "High Consumption"
        saving_percentage = 0.20
        recommendations = [
            "Reduce AC temperature by 1–2°C.",
            "Turn off unused appliances.",
            "Use natural lighting during daytime.",
            "Shift heavy appliance usage to off-peak hours.",
            "Enable Smart Power Mode.",
        ]

    elif predicted_consumption >= 2:
        category = "Medium Consumption"
        saving_percentage = 0.15
        recommendations = [
            "Monitor appliance usage.",
            "Use energy-efficient appliances.",
            "Avoid unnecessary lighting.",
            "Schedule washing machine usage during off-peak hours.",
        ]

    else:
        category = "Low Consumption"
        saving_percentage = 0.08
        recommendations = [
            "Maintain the current usage pattern.",
            "Continue using energy-saving devices.",
        ]

    estimated_saving = predicted_consumption * saving_percentage
    optimized_consumption = predicted_consumption - estimated_saving
    money_saved = estimated_saving * ELECTRICITY_COST_PER_KWH

    return {
        "category": category,
        "predicted_consumption_kwh": predicted_consumption,
        "optimized_consumption_kwh": optimized_consumption,
        "energy_saved_kwh": estimated_saving,
        "estimated_money_saved": money_saved,
        "recommendations": recommendations,
    }
