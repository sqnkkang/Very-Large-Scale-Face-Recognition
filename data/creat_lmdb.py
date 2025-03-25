import os
import lmdb
import cv2
from tqdm import tqdm

class LMDB:
    def __init__(self, lmdb_path):
        self.map_size = 500 * 1024 * 1024
        self.env = lmdb.open(lmdb_path, map_size=self.map_size)
        self.kv = {}
        self.buf_size = 1000
    def put(self, k, v):
        if k in self.kv:
            print('%s is already in the db.' % k)
        else:
            self.kv[k] = v
            if len(self.kv) >= self.buf_size:
                self.commit()
    def commit(self):
        if len(self.kv) > 0:
            txn = self.env.begin(write=True)
            for k, v in self.kv.items():
                try:
                    txn.put(k, v)
                except lmdb.MapFullError:
                    txn.abort()
                    self.map_size = self.map_size * 3 // 2
                    self.env.set_mapsize(self.map_size)
                    self.commit()
                    return
            try:
                txn.commit()
            except lmdb.MapFullError:
                txn.abort()
                self.map_size = self.map_size * 3 // 2
                self.env.set_mapsize(self.map_size)
                self.commit()
                return
            self.kv = {}
            txn.abort()
    def close(self):
        self.commit()
        self.env.close()

def make_lmdb(image_src_dir, path_to_lmdb, db_name):
    dirs = os.listdir(image_src_dir)
    db = LMDB(path_to_lmdb)
    kv = open(os.path.join(path_to_lmdb, '%s_kv.txt' % db_name), 'w')
    next_label = 0
    for d in tqdm(dirs, desc="Processing", ascii=True, leave=True):
        sub_dir = os.path.join(image_src_dir, d)
        if os.path.isdir(sub_dir):
            images = os.listdir(sub_dir)
            files = []
            for fn in images:
                ext = os.path.splitext(fn)[1].lower()
                if ext in ('.jpg', '.png', '.bmp', '.jpeg'):
                    files.append(os.path.join(sub_dir, fn))
            if len(files) > 0:
                for j, path in enumerate(files):
                    cv_img = cv2.imread(path)
                    cv_img = cv2.resize(cv_img, (224, 224))
                    _, img_bytes = cv2.imencode('.jpg', cv_img)
                    img_bytes = img_bytes.tobytes()
                    key = '%s_%d_%d' % (db_name, next_label, j)
                    db.put(key.encode('utf-8'), img_bytes)
                    kv.write('%s %d\n' % (key, next_label))
                next_label += 1
    db.close()
    kv.close()

if __name__ == '__main__':
    image_src_dir = './data'
    path_to_lmdb = './lmdb'
    make_lmdb(image_src_dir, path_to_lmdb, 'train')
    print("创建成功")
