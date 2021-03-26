from influxdb import InfluxDBClient
# 首先连接influxdb
client = InfluxDBClient(host='141.164.56.99', port=8086, username='myuser', password='mypass' ssl=True, verify_ssl=True)
# 创建数据库
client.create_database('database_name')
# 查询数据库
client.get_list_database()
