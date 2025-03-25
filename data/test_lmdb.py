import lmdb
import cv2
import numpy as np

def read_lmdb(lmdb_path, kv_file_path):
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
        img = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
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
