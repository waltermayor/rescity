import MySQLdb

def register_count():

        db = MySQLdb.connect(host="localhost",    # your host, usually localhost
                     user="root",         # your username
                     passwd="root",  # your password
                     db="accident_patterns")        # name of the data base

        # you must create a Cursor object. It will let
        #  you execute all the queries you need
        cur = db.cursor()

        # Use all the SQL you like
        sel
        cur.execute("INSERT INTO accident_events(id,time,probability,accident,id_pattern,id_vehicles,id_location) VALUES (NULL, CURRENT_TIMESTAMP,%s, %s,%s, %s,%s)",(str(100),str(0),str(1),str(2),str(3)))
        #db.commit()
        try:
                #cur.execute("INSERT INTO `conteo` (`id`, `time`, `count`) VALUES (NULL, CURRENT_TIMESTAMP, '"+str(count)+"');")
                #cur.execute("INSERT INTO accident_events(id,time,probability,accident,id_pattern,id_vehicles,id_location) VALUES (NULL, CURRENT_TIMESTAMP,%s, %s,%s, %s,%s)",(str(100),str(0),str(1),str(2),str(3)))

                db.commit()
        except:
                db.rollback()

if __name__ == '__main__':

    register_count()
