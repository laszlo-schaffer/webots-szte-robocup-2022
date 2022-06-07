"""ks0193_base controller."""

# You may need to import some classes of the controller module. Ex:
from controller import Robot
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
    angle = -np.arctan2(ay , az)  # Radial rotation angle calculation formula; the negative sign is direction processing
    return angle, gx
  
def setSpeed(motors, values=[0, 0]):
    motors[0].setVelocity(values[0])
    motors[1].setVelocity(values[1])

# -----------------------------------------------------

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)

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
setpoint = 0.0 #  rad    
pid = PID(25, 200, 0.0, setpoint=setpoint)
while robot.step(timestep) != -1:
    # Wait after simulation start for the robot to bounce (Webots physics)
    if robot.getTime() > 0.02:
        # Process sensor data here.
        angle_rad, angle_speed = process_imu(acc, gyro)
        # prototype Kalman filter
        # angle, angle_speed = KF.estimate(angle, angle_speed)      
        angle = angle_rad * (180/np.pi)
        if PLOTTING:
            plot_out.append(angle)

        error = setpoint - angle_rad
        control = pid(error)
        # saturate to [-100,100] interval
        speed_out = np.clip([control,control], -16.76, 16.76)
        # debug
        #print(control, [angle_rad, angle, angle_speed], speed_out)
        print(angle)
        if np.abs(angle) >= 45:
            # stop motors if robot falls over
            speed_out = [0, 0]
            setSpeed(motors, speed_out)
            print("break")
            break
        # command motors 
        setSpeed(motors, speed_out)

if PLOTTING:
    ser = Series(plot_out)
    ser.plot()
    plt.show()
        

# Enter here exit cleanup code.
