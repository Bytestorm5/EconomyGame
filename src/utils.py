import math

def age_decay(base: float, age: float, peak: float, steepness: float, floor: float) -> float:
    if age <= peak:
        return base
    decay = 1.0 / (1.0 + math.exp(steepness * (age - peak)))
    return floor + (base - floor) * decay
