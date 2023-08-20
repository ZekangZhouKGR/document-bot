from pymongo import MongoClient

# 连接到MongoDB数据库
client = MongoClient('mongodb://mongodb:27017/')
db = client['code_templates']  # 替换为您的数据库名称
collection = db['templates']  # 替换为您的集合名称

# 要插入的模板数据
template = {
    "template_name": "User Registration",
    "description": "A code template for user registration",
    "language": "Python",
    "tags": ["registration", "user"],
    "domain": "Web Development",
    "code": """Some code here..."""
}

# 插入模板数据
result = collection.insert_one(template)
print("已插入模板，对象ID为: ", result.inserted_id)

# 关闭数据库连接
client.close()