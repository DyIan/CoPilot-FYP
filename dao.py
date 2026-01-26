import psycopg2

class DAO:
    def __init__(self, broker):

        self.connection = psycopg2.connect(database="fyp", user="postgres", password="Malahide12321", host="localhost", port="5432")
        self.cursor = self.connection.cursor()

        self.long = None
        self.lat =None

        self.broker = broker
        broker.subscribe("gps", self.gps_callback)
        

    def gps_callback(self, coords):
        """ Game loop publishes GPS coords every second"""
        x, y = coords

        # Since CARLA doesnt do real coords we will just make up our own based on area
        # Back Half Medium 50km/h
        if -120 <= x <= -62 and -76.0 <= y <= 146.0:
            self.long = -6.2395
            self.lat = 53.46446


        # Middle Fast 90 hm/h
        elif -63 <= x <= -34.5 and -76.0 <= y <= 146.0:
            self.long = -6.202502
            self.lat = 53.469158

        # Top and neighbourhood 30km/h
        else:
            self.long = -6.2522
            self.lat = 53.4603
        
        # Check the db now for speed limit in this area
        self.update_speed_limit()

    def update_speed_limit(self):
        """ Takes self.long/lat and checks database for nearest speed limit from those coords and then publishes to 'speed_limit' """
        try:
            self.cursor.execute("SELECT * FROM find_speed_limit (%s, %s)", (self.long, self.lat))
            result = self.cursor.fetchone()
        
            if result:
                #print(result)
                speed_limit = float(result[1])
                self.broker.publish("speed_limit", speed_limit)
        except Exception as e:
            print("DAO error: ", e)
        