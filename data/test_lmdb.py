import lmdb
import cv2
import numpy as np
from datum_def_pb2 import Datum

def read_lmdb(lmdb_path, kv_file_path):
    '''
    打开 lmdb 数据库，遍历我们的 kv 文件读取出键值然后直接查找
    '''
    env = lmdb.open(lmdb_path, readonly=True)
    txn = env.begin()
    with open(kv_file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        key, label = line.strip().split()
        value = txn.get(key.encode('utf-8'))
        if value is None:
            print(f"Key {key} is not in lmdb")
            continue
        '''
        反序列化 datum 对象，解码图片数据，进行展示输出
        '''
        datum = Datum()
        datum.ParseFromString(value)
        img_data = datum.data
        img = cv2.imdecode(np.frombuffer(img_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        '''
        显示图片和对应的标签
        '''
        cv2.imshow(f"Image {key} (Label: {label})", img)
        print(f"Key: {key}, Label: {label}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    env.close()
if __name__ == '__main__':
    lmdb_path = './lmdb'
    kv_file_path = './lmdb/train_kv.txt'
    read_lmdb(lmdb_path, kv_file_path)
    print("测试完成")
