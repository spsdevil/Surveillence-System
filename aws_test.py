import tinys3

S3_ACCESS_KEY = "balcotoday"
S3_SECRET_KEY = "Balc0t0day"

conn = tinys3.Connection(S3_ACCESS_KEY,S3_SECRET_KEY,tls=True,endpoint='s3-us-west-1.amazonaws.com')
print("connected")
print("uploading")

f = open('data.txt','rb')
conn.upload('data.txt',f,'aontrail')
print("DONE")