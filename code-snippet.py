import psycopg2

mydb = psycopg2.connect(
  host="192.168.40.6",
  user="postgres",
  password="postgres123",
  database="bot_crypto"
)

mycursor = mydb.cursor()

mydb.set_session(autocommit=True)


mycursor.execute("SELECT * FROM crypto")

print(mycursor.fetchall())

mycursor.close()
mydb.close()