"""
tick_time.py — Centralized tick-to-time conversions.

1 tick = 1 game-minute.
All constants are in ticks unless otherwise noted.
To reconfigure simulation speed, change TICKS_PER_MINUTE and derived constants update automatically.
"""

TICKS_PER_MINUTE: int = 1          # 1 real tick = 1 game-minute (configurable)
TICKS_PER_HOUR:   int = TICKS_PER_MINUTE * 60
TICKS_PER_DAY:    int = TICKS_PER_HOUR * 24          # 1,440
TICKS_PER_WEEK:   int = TICKS_PER_DAY * 7            # 10,080
TICKS_PER_MONTH:  int = TICKS_PER_DAY * 30           # 43,200
TICKS_PER_YEAR:   int = TICKS_PER_DAY * 365          # 525,600

# Derived: fractional year per tick (for age increment)
YEARS_PER_TICK:   float = 1.0 / TICKS_PER_YEAR       # ~0.0000019

# Entrepreneurship
VENTURE_COOLDOWN_TICKS:      int = TICKS_PER_MONTH * 1   # 30-day cooldown after failed/launched venture
ENTREPRENEURSHIP_EVAL_TICKS: int = TICKS_PER_DAY         # evaluate once per game-day

# Loans / banking
LOAN_PAYMENT_PERIOD_TICKS: int = TICKS_PER_MONTH         # payment due every 30 game-days
LOAN_LIQUIDATION_WINDOW_TICKS: int = TICKS_PER_MONTH * 3 # 90 game-days to liquidate before seizure

# Population lifecycle
BIRTH_CHECK_INTERVAL_TICKS: int = TICKS_PER_DAY          # check for births once per game-day
MAX_POPULATION: int = 200                                  # soft cap; no births above this
BASE_BIRTH_CHANCE_PER_DAY: float = 0.02                   # 2% chance per day of a new citizen spawning

# Education
EDUCATION_PROMOTION_TICKS: int = TICKS_PER_DAY * 90      # 90 game-days of work to gain education in a domain

# Career progression
CAREER_EVAL_INTERVAL_TICKS: int = TICKS_PER_DAY * 7      # weekly career evaluation
PROMOTION_MIN_TICKS: int = TICKS_PER_DAY * 60            # 60 game-days before eligible for promotion
