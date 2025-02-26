import numpy as np
from airplane.grumman import Grumman

class ReducedGrumman(Grumman):
    ####################################
    ### Grumman American AA-1 Yankee ###
    ####################################
    """Class for simplified airplane state and dynamics"""

    # NOTE: Commands as seperate objects? e.g. bank.rotate(airplane),
    #       throttle.accelerate(airplane), etc.
    # NOTE: Use of α instead of Cl?

    def __init__(self):
        super().__init__()
        ##########################
        ### Airplane variables ###
        ##########################
        self.flight_path_angle = np.zeros(10000, dtype=np.float32)                  # Flight Path Angle  (γ)  [rad]
        self.airspeed_norm = np.ones_like(self.flight_path_angle, dtype=np.float32) # Air Speed  (V/Vs)  [1]
        self.bank_angle = 0.0  # Bank Angle  (μ)  [rad]
        # previous commands
        self.last_c_lift = 0.0
        self.last_bank_rate = 0.0
        self.last_throttle = 0.0

    def command_airplane(self, c_lift, bank_rate, delta_throttle):
        self.last_c_lift = c_lift
        self.last_bank_rate = bank_rate
        self.last_throttle = delta_throttle
        
        c_drag = self._cd_from_cl(c_lift)

        # V_dot = - g sin γ - 0.5 * (ρ S V^2 CD / m) +  (thrust force / m) 
        airspeed_dot = - self.GRAVITY * np.sin(self.flight_path_angle) - 0.5 * self.AIR_DENSITY * (
                self.WING_SURFACE_AREA / self.MASS) * (self.airspeed_norm * self.STALL_AIRSPEED) ** 2 * c_drag \
                       + (self.THROTTLE_LINEAR_MAPPING * delta_throttle / self.MASS)
        
        # γ_dot = 0.5 * (ρ S V CL cos µ / m) - g cos γ / V
        flight_path_angle_dot = 0.5 * self.AIR_DENSITY * (self.WING_SURFACE_AREA / self.MASS) * (
                self.airspeed_norm * self.STALL_AIRSPEED) * c_lift * np.cos(self.bank_angle) \
                              - (self.GRAVITY / (self.airspeed_norm * self.STALL_AIRSPEED)) * np.cos(
            self.flight_path_angle)

        # μ_dot = μ_dot_commanded
        bank_angle_dot = bank_rate

        
        self.airspeed_norm = self._update_state_from_derivative(self.airspeed_norm, airspeed_dot / self.STALL_AIRSPEED)
        self.flight_path_angle = self._update_state_from_derivative(self.flight_path_angle, flight_path_angle_dot)
        self.bank_angle = self._update_state_from_derivative(self.bank_angle, bank_angle_dot)

    def reset(self, flight_path_angle, airspeed_norm, bank_angle):
        self.flight_path_angle = flight_path_angle
        self.airspeed_norm = airspeed_norm
        self.bank_angle = bank_angle