import numpy as np


class Grumman:
    ####################################
    ### Grumman American AA-1 Yankee ###
    ####################################
    """Base class for airplane parameters"""

    def __init__(self):
        ######################
        ### Sim parameters ###
        ######################
        self.TIME_STEP = 0.1
        self.GRAVITY = 9.81
        self.AIR_DENSITY = 1.225  # Density  (ρ)  [kg/m3]

        ###########################
        ### Airplane parameters ###
        ###########################
        # Aerodynamic model: CL coefficients
        self.CL_0 = 0.41
        self.CL_ALPHA = 4.6983
        self.CL_ELEVATOR = 0.361
        self.CL_QHAT = 2.42
        # Aerodynamic model: CD coefficients
        self.CD_0 = 0.0525
        self.CD_ALPHA = 0.2068
        self.CD_ALPHA2 = 1.8712
        # Aerodynamic model: Cm coefficients
        self.CM_0 = 0.076
        self.CM_ALPHA = -0.8938
        self.CM_ELEVATOR = -1.0313
        self.CM_QHAT = -7.15
        # Aerodynamic model: Cl coefficients
        self.Cl_BETA = -0.1089
        self.Cl_PHAT = -0.52
        self.Cl_RHAT = 0.19
        self.Cl_AILERON = -0.1031
        self.Cl_RUDDER = 0.0143
        # Physical model
        self.MASS = 697.18  # Mass  (m)  [kg]
        self.WING_SURFACE_AREA = 9.1147  # Wing surface area  (S)  [m2]
        self.CHORD = 1.22  # Chord  (c)  [m]
        self.WING_SPAN = 7.46  # Wing Span  (b)  [m]
        self.I_XX = 808.06   # Inertia  [Kg.m^2]
        self.I_YY = 1011.43  # Inertia  [Kg.m^2]
        self.ALPHA_STALL = np.deg2rad(15)  # Stall angle of attack  (αs)  [rad]
        self.ALPHA_NEGATIVE_STALL = np.deg2rad(-7)  # Negative stall angle of attack  (αs)  [rad]
        self.CL_STALL = self.CL_0 + self.CL_ALPHA * self.ALPHA_STALL
        self.CL_REF = self.CL_STALL 
        # self.STALL_AIRSPEED = 32.19  # Stall air speed  (Vs)  [m/s]
        self.STALL_AIRSPEED = np.sqrt(self.MASS * self.GRAVITY / (0.5 * self.AIR_DENSITY * \
                                    self.WING_SURFACE_AREA * self.CL_REF))    # Stall air speed  (Vs)  [m/s]
        print(f"STALL_AIRSPEED: {self.STALL_AIRSPEED}")
        
        self.MAX_CRUISE_AIRSPEED = 2 * self.STALL_AIRSPEED  # Maximum air speed  (Vs)  [m/s]

        # Throttle model
        self.THROTTLE_LINEAR_MAPPING = None
        self._initialize_throttle_model()

    def _update_state_from_derivative(self, value_to_update, value_derivative):
        value_to_update += self.TIME_STEP * value_derivative
        return value_to_update

    def _alpha_from_cl(self, c_lift):
        alpha = (c_lift - self.CL_0) / self.CL_ALPHA
        return alpha

    def _cl_from_lift_force_and_speed(self, lift_force, airspeed):
        cl = 2 * lift_force / (self.AIR_DENSITY * self.WING_SURFACE_AREA * airspeed ** 2)
        return cl

    def _cl_from_alpha(self, alpha, elevator, q_hat):
        # TODO: review model
        if alpha <= self.ALPHA_NEGATIVE_STALL:
            c_lift = self.CL_0 + self.CL_ALPHA * self.ALPHA_NEGATIVE_STALL
        elif alpha >= self.ALPHA_STALL:
            # Stall model: Lift saturation
            c_lift = self.CL_0 + self.CL_ALPHA * self.ALPHA_STALL
            # Stall model: Lift reduction with opposite slope
            # c_lift = - self.CL_ALPHA * alpha + self.CL_0 + 2 * self.CL_ALPHA * self.ALPHA_STALL
        else:
            c_lift = self.CL_0 + self.CL_ALPHA * alpha + self.CL_ELEVATOR * elevator + self.CL_QHAT * q_hat
        return c_lift

    def _lift_force_at_speed_and_cl(self, airspeed, lift_coefficient):
        return 0.5 * self.AIR_DENSITY * self.WING_SURFACE_AREA * airspeed ** 2 * lift_coefficient

    def _cd_from_alpha(self, alpha):
        c_drag = self.CD_0 + self.CD_ALPHA * alpha + self.CD_ALPHA2 * (alpha ** 2)
        return c_drag

    def _cd_from_cl(self, c_lift):
        c_drag = self._cd_from_alpha(self._alpha_from_cl(c_lift))
        return c_drag

    def _drag_force_at_speed_and_cd(self, airspeed, drag_coefficient):
        return 0.5 * self.AIR_DENSITY * self.WING_SURFACE_AREA * airspeed ** 2 * drag_coefficient

    def _drag_force_at_cruise_speed(self, airspeed):
        cruise_lift_force = self.MASS * self.GRAVITY
        cruise_cl = self._cl_from_lift_force_and_speed(cruise_lift_force, airspeed)
        alpha = self._alpha_from_cl(cruise_cl)
        cruise_cd = self._cd_from_alpha(alpha)
        drag_force = self._drag_force_at_speed_and_cd(airspeed, cruise_cd)
        return drag_force

    def _rolling_moment_coefficient(self, beta, p_hat, r_hat, aileron, rudder):
        c_rolling_moment = self.Cl_BETA * beta + self.Cl_PHAT * p_hat + self.Cl_RHAT * r_hat + \
                 self.Cl_AILERON * aileron + self.Cl_RUDDER * rudder
        return c_rolling_moment

    def _rolling_moment_at_speed_and_cl(self, airspeed, rolling_moment_coefficient):
        return 0.5 * self.AIR_DENSITY * self.WING_SURFACE_AREA * self.WING_SPAN * airspeed ** 2 * rolling_moment_coefficient

    def _pitching_moment_coefficient(self, alpha, elevator, q_hat):
        c_pitch_moment = self.CM_0 + self.CM_ALPHA * alpha + self.CM_ELEVATOR * elevator + self.CM_QHAT * q_hat
        return c_pitch_moment

    def _pitching_moment_at_speed_and_cm(self, airspeed, pitching_moment_coefficient):
        return 0.5 * self.AIR_DENSITY * self.WING_SURFACE_AREA * self.CHORD * airspeed ** 2 * pitching_moment_coefficient

    def _initialize_throttle_model(self, ):
        # Throttle model: Thrust force = Kt * δ_throttle
        # Max Thrust -> Kt * 1 = Drag(V=Vmax) -> Kt = 0.5 ρ S (Vmax)^2 CD
        # δ_throttle = 1.0  -> Max Cruise speed: V' = Vmax  ->  V_dot = 0 = Thrust Force - Drag Force
        self.THROTTLE_LINEAR_MAPPING = self._drag_force_at_cruise_speed(self.MAX_CRUISE_AIRSPEED)

    def _thrust_force_at_throttle(self, throttle):
        thrust_force = self.THROTTLE_LINEAR_MAPPING * throttle
        return thrust_force