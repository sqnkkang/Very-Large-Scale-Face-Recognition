import os
import lmdb
import cv2
from datum_pb2 import Datum

class LMDB:
    '''
    初始化函数，传进来想存储的 lmdb 的地址，构建 map_size 大小的数据库
    同时构建一个 kv 字典，存储数据库里面的键值对，定义缓存空间的大小
    '''
    def __init__(self, lmdb_path):
        self.map_size = 500 * 1024 * 1024
        self.env = lmdb.open(lmdb_path, map_size=self.map_size)
        self.kv = {}
        self.buf_size = 1000
    '''
    重写一下 put，我们现在的数据库是动态的，有一个缓存的空间，当发现存储一个数据现在已经在 kv
    字典里面，打印出提示语句，否则把当前的键值对加入到 kv 文件里面，发现缓存满了进行一次提交
    '''
    def put(self, k, v):
        if k in self.kv:
            print('%s 已经存在 db 数据库里面' % k)
        else:
            self.kv[k] = v
            if len(self.kv) >= self.buf_size:
                self.commit()
    '''
    重写 commit，发现当前的 kv 文件里面有值并且调用 commit 函数，证明需要提交
    打开当前的环境进行写入即可，要是抛出异常，发现当前的空间太小，终止我们的写入，扩充空间，重新设置数据库的大小
    然后再次执行当前的函数，将剩余的部分 commit 进去
    
    写完之后尝试提交，抛出异常的话执行一样的操作，清空临时的 kv 键值对字典
    '''
    def commit(self):
        if len(self.kv) > 0:
            txn = self.env.begin(write=True)
            for k, v in self.kv.items():
                try:
                    txn.put(k, v)
                except lmdb.MapFullError:
                    txn.abort()
                    self.map_size = self.map_size * 3 // 2  # double map size and recommit
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
    '''
    提交一下当前的键值对，之后关闭数据库
    '''
    def close(self):
        self.commit()
        self.env.close()
'''
图片的文件需要按照下面的形式
image_root
    sub_dir1
         --1.jpg
         --2.jpg
    sub_dir2
         --1.jpg
         --2.jpg
         --3.jpg
    ...
    sub_dirn
         --1.jpg
         --2.jpg
         --3.jpg
每个 sub_dir 是同一个类别
'''
def make_lmdb(image_src_dir, path_to_lmdb, db_name):
    '''
    得到当前的根目录下面的所有 sub_dir，然后构建数据库，以及构建 kv 文件
    '''
    dirs = os.listdir(image_src_dir)
    db = LMDB(path_to_lmdb)
    kv = open(os.path.join(path_to_lmdb, '%s_kv.txt' % db_name), 'w')
    next_label = 0
    for d in dirs:
        '''
        构建完整的 sub_dir 路径
        '''
        sub_dir = os.path.join(image_src_dir, d)
        '''
        存在的话构建图片地址列表
        '''
        if os.path.isdir(sub_dir):
            images = os.listdir(sub_dir)
            files = []
            for fn in images:
                '''
                遍历整个图片的文件，提取文件的拓展名将其转化为小写，这样的话就能判断他的图片类型
                '''
                ext = os.path.splitext(fn)[1].lower()
                if ext in ('.jpg', '.png', '.bmp', '.jpeg'):
                    '''
                    将其绝对路径加到我们的 files 列表里面
                    '''
                    files.append(os.path.join(sub_dir, fn))
            '''
            根据路径加载出来图片，然后构建 Datum 对象，对我们的图片信息进行序列化操作，然后存储到 lmdb 数据库里面
            更新我们的 kv 文件，存储结束关闭数据库和 kv 文件
            '''
            if len(files) > 0:
                for j, path in enumerate(files):
                    cv_img = cv2.imread(path)
                    cv_img = cv2.resize(cv_img, (224, 224))
                    key = '%s_%d_%d' % (db_name, next_label, j)
                    datum = Datum()
                    '''
                     从元组中提取第二个元素，即编码后的图像数据（buffer），cv2.imencode('.jpg', cv_img) 返回 (True, buffer)，则 [1] 提取 buffer
                    '''
                    datum.data = cv2.imencode('.jpg', cv_img)[1].tobytes()
                    db.put(key.encode('utf-8'), datum.SerializeToString())
                    kv.write('%s %d\n' % (key, next_label))
                next_label += 1
    db.close()
    kv.close()

if __name__ == '__main__':
    '''
    建议按照我这个路径存储数据库的文件，直接执行当前的文件即可
    '''
    image_src_dir = './data'
    path_to_lmdb = './lmdb'
    make_lmdb(image_src_dir, path_to_lmdb, 'train')
    print("创建成功")
