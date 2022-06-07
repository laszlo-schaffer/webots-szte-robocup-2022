"""ks0193_base controller."""

# You may need to import some classes of the controller module. Ex:
from controller import Robot, Keyboard
import numpy as np
from simple_pid import PID

PLOTTING = False  # plotting angle data
if PLOTTING:
    from pandas import Series
    import matplotlib.pyplot as plt

# prototype Kalman filter based on
# https://wiki.keyestudio.com/Ks0193_keyestudio_Self-balancing_Car
class Kalman:
    def __init__(self):
        self.Q_angle = 0.001  #Covariance of gyroscope noise
        self.Q_gyro = 0.003    #Covariance of gyroscope drift noise
        self.R_angle = 0.5    #Covariance of accelerometer
        self.C_0 = 1
        self.dt = 0.005 #The value of dt is the filter sampling time.
        self.K1 = 0.05 # a function containing the Kalman gain is used to calculate the deviation of the optimal estimate.
        self.q_bias = 0.0
        self.angle=0.0
        
        self.Pdot = np.zeros((4,1))
        self.P = np.array([[1,0], [0,1]])
    
    def estimate(self, angle_m, gyro_m):
        self.angle += (gyro_m - self.q_bias) * self.dt          #Prior estimate
        self.angle_err = angle_m - self.angle
        
        self.Pdot[0] = self.Q_angle - self.P[0][1] - self.P[1][0]    #Differential of azimuth error covariance
        self.Pdot[1] = -self.P[1][1]
        self.Pdot[2] = -self.P[1][1]
        self.Pdot[3] = self.Q_gyro
          
        self.P[0][0] += self.Pdot[0] * self.dt    #The integral of the covariance differential of the prior estimate error
        self.P[0][1] += self.Pdot[1] * self.dt
        self.P[1][0] += self.Pdot[2] * self.dt
        self.P[1][1] += self.Pdot[3] * self.dt
          
        #Intermediate variable of matrix multiplication
        self.PCt_0 = self.C_0 * self.P[0][0]
        self.PCt_1 = self.C_0 * self.P[1][0]
        #Denominator
        self.E = self.R_angle + self.C_0 * self.PCt_0
        #Gain value
        self.K_0 = self.PCt_0 / self.E
        self.K_1 = self.PCt_1 / self.E
          
        self.t_0 = self.PCt_0;  #Intermediate variable of matrix multiplication
        self.t_1 = self.C_0 * self.P[0][1]
          
        self.P[0][0] -= self.K_0 * self.t_0   #Posterior estimation error covariance 
        self.P[0][1] -= self.K_0 * self.t_1
        self.P[1][0] -= self.K_1 * self.t_0
        self.P[1][1] -= self.K_1 * self.t_1
          
        self.q_bias += self.K_1 * self.angle_err   #Posterior estimation
        self.angle_speed = gyro_m - self.q_bias  # The differential value of the output value; work out the optimal angular velocity
        self.angle += self.K_0 * self.angle_err #  Posterior estimation; work out the optimal angle
        return self.angle, self.angle_speed
    
# Helper functions -----------------------------------
def process_imu(acc, gyro):
    rad_values = np.array(gyro.getValues()) #  rad/sec
    [gx, gy, gz] = rad_values*(180/np.pi) #  deg/sec
    [ax, ay, az] = acc.getValues()
    #print(ax, ay, az)
    angle = -np.arctan2(ay , az)  # Radial rotation angle calculation formula; the negative sign is direction processing
    return angle, gx
  
def setSpeed(motors, values):
    motors[0].setVelocity(float(values[0]))
    motors[1].setVelocity(float(values[1]))

# -----------------------------------------------------

# create the Robot instance.
robot = Robot()
keyboard = Keyboard()
# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())
keyboard.enable(timestep)

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)

weight = robot.getDevice('weight_position')
debug_device = robot.getDevice('imu')
debug_device.enable(timestep)

#Set up motors
motorL = robot.getDevice('motor_L')
motorL.setPosition(float('inf'))
motorL.setVelocity(0)
motorR = robot.getDevice('motor_R')
motorR.setPosition(float('inf'))
motorR.setVelocity(0)
motors = [motorL, motorR]

#Set up Encoders
motorLEncoder = robot.getDevice("motor_L_encoder")
motorLEncoder.enable(timestep)
motorREncoder = robot.getDevice("motor_R_encoder")
motorREncoder.enable(timestep)

# Get reference value of encoder
lEncRef = motorLEncoder.getValue()
rEncRef = motorREncoder.getValue()

# Set up MPU6050
acc = robot.getDevice('mpu6050_acc')
gyro = robot.getDevice('mpu6050_gyro')
gyro.enable(timestep)
acc.enable(timestep)


# Main loop:
# - perform simulation steps until Webots is stopping the controller

# prototype Kalman filter
KF = Kalman()

speed_out = [0, 0]
# ---------------------------------
if PLOTTING:
    plot_out = []
    
# PID controller, requires: simple_pid package | pip install simple_pid
#angle_setpoint = 45/(180/np.pi) #  rad
angle_setpoint = -13.0 #  deg
pos_setpoint = 0.0
# simple_pid
#kp_angle = 5
#ki_angle = 2.5
#kd_angle = -2.5

#pid = PID(kp_angle, ki_angle, kd_angle, setpoint=angle_setpoint)
#pid.sample_time = 1/timestep

# manual pid
kp_angle = 0.7  #0.4 #34
ki_angle = None
kd_angle = 0.25  #0.62

kp_speed = 5.5 #3.6
ki_speed = 0.25 #0.08
kd_speed = None

# Weight position: ks0193 -> children -> threaded rod ->
# children -> SliderJoint -> device -> jointParameters -> position

weightPos = 0.0
#weight.setPosition(weightPos) # Max 0.1349
# Manual mode params
automatic = True  # True - Automatic
kspeed = 4*np.pi # manual speed
WHEEL_RADIUS = 0.03335 # meters
MAXVELOCITY = 16.75  # equals 160 rpm
STARTTIME = 0.02
while robot.step(timestep) != -1:
    y,p,r = np.array(debug_device.getRollPitchYaw())
    encL_rad = motorLEncoder.getValue()
    encR_rad = motorREncoder.getValue()
    angle_rad, angle_speed = process_imu(acc, gyro)
    posL = encL_rad * WHEEL_RADIUS
    posR = encR_rad * WHEEL_RADIUS
    speedL = motors[0].getVelocity()
    speedR = motors[1].getVelocity()
    speed = np.round((speedL + speedR)/2, 2)
    pos = np.round((posL + posR)/2, 2)
    print("Pitch:", np.round(p*(180/np.pi), 2), "pos:", pos, "speed", speed)
    if not automatic:
        # Deal with the pressed keyboard key.
        key1 = keyboard.getKey()
        key2 = keyboard.getKey()
        keys = [key1, key2]
        if ord('W') in keys and ord('A') in keys:
            speed_out = [0.5*kspeed, kspeed]
        elif ord('W') in keys and ord('D') in keys:
            speed_out = [kspeed, 0.5*kspeed]
        elif ord('S') in keys and ord('A') in keys:
            speed_out = [-0.5*kspeed, kspeed]
        elif ord('S') in keys and ord('D') in keys:
            speed_out = [kspeed, -0.5*kspeed]
        elif ord('W') in keys:
            speed_out = [kspeed, kspeed]
        elif ord('S') in keys:
            speed_out = [-1*kspeed, -1*kspeed]
        elif ord('A') in keys:
            speed_out = [0, kspeed]
        elif ord('D') in keys:
            speed_out = [kspeed, 0]
        elif ord('E') in keys:
            weightPos += 0.01
            weightPos = np.clip(weightPos, 0.0, 0.1349)
            weight.setPosition(weightPos) # Max 0.1349
        elif ord('Q') in keys:
            weightPos -= 0.01
            weightPos = np.clip(weightPos, 0.0, 0.1349)
            weight.setPosition(weightPos) # Max 0.1349
        else:
            speed_out = [0, 0]
        setSpeed(motors, speed_out)
    
    if automatic:
        # Wait after simulation start for the robot to bounce (Webots physics)
        if robot.getTime() > STARTTIME:
            # Process sensor data here.
            angle = angle_rad * (180/np.pi)
            # prototype Kalman filter
            #angle, angle_speed = KF.estimate(angle, angle_speed)
            print(angle)
            
            # Based on Accelerometer and Gyro:
            angle_error = angle_setpoint - angle
            # Based on InertialUnit - Pitch:
            #angle_error = angle_setpoint - p*(180/np.pi)
            pos_error = pos_setpoint - pos
            PD_value = kp_angle * angle_error + kd_angle * angle_speed
            PI_value = ki_speed * pos_error + kp_speed * (pos_setpoint-speed)
            control = -1*PD_value + -1*PI_value
            
            if PLOTTING:
                #plot_out.append(angle)
                plot_out.append(angle_error)
            
            if np.abs(p*(180/np.pi)) >= 45:
                # stop motors if robot falls over
                speed_out = [0, 0]
                setSpeed(motors, speed_out)
                print("break")
                break
            # saturate
            speed_out = np.clip([control,control], -MAXVELOCITY, MAXVELOCITY)

            # debug
            #to_print = np.array([angle, angle_speed, angle_error, pos_error, speed, PD_value, PI_value, control])
            #print(np.round(to_print, 2))
            
            # command motors 
            setSpeed(motors, speed_out)
        else:
            # No control at the beginning
            angle = angle_rad * (180/np.pi)
            print("//", angle, "//")

    

if PLOTTING:
    ser = Series(plot_out)
    ser.plot()
    plt.show()
        

# Enter here exit cleanup code.
