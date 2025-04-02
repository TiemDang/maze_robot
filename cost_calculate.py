import math



class calculate_cost :
    def __init__(self, x, y, theta, x_dot, y_dot, sensor_valid, distance):
        self.x = x
        self.y = y
        self.theta = theta
        
        self.x_dot = x_dot
        self.y_dot = y_dot
        self.sensor_valid = sensor_valid
        self.distance = distance

    def euclid_score(self):
        euclid_score = (math.sqrt((self.x - 450)**2 + (self.y - 900)**2 )) * 100 + math.sqrt((self.theta - (-math.pi/2))**2) * 500
        return euclid_score
    
    
    def sensor_score(self):
        num_sensor = 0
        for distan in self.distance :
            if self.distance[0] > 100 or self.distance[6] > 100 or self.distance[7] > 100 :
                num_sensor = num_sensor + 1


        if num_sensor == 0:
            sensor_score = 600
        elif num_sensor > 1 :
            sensor_score = (num_sensor - 1) * -300 
        return sensor_score


    def distance_score(self):
        mid1_4 = math.sqrt((self.distance[4] - self.distance[1])**2)
        mid2_5 = math.sqrt((self.distance[2] - self.distance[5])**2)
        
        distance_score = (mid1_4 + mid2_5) * 2
        return distance_score
    
