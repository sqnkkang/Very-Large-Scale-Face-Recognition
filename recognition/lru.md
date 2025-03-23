# LRU策略

## 1 >> 创建节点信息

- 创建双向链表的节点类，分别代表前一个结点，后一个节点，节点当前的键值和，允许被弱引用，弱引用不会增加对象的引用计数，适合用于缓存等场景，避免内存泄漏。

```python
class LinkNode(object):
    __slots__ = 'prev', 'next', 'key', 'value', '__weakref__'
```

## 2 >> 创建操作信息

- 记录 LRU 的缓存操作，操作类型有三种 Add Overflow Get， op_type 记录的就是操作类型，下面介绍了操作中涉及的前一个结点，后一个节点，和移除的节点，最后记录的是操作的键值对，为了我们的 LRU 的回滚操作。

```python
class Op(object):
    def __init__(self, op_type, prev_node=None, next_node=None, removed_node=None, key=None, value=None):
        self.op_type = op_type
        self.prev_node = prev_node
        self.next_node = next_node
        self.removed_node = removed_node
        self.kv = (key, value)
```

## 3 >> 创建 LRU 类

- \_\_init\_\_ 初始化函数，记录了缓存的核心逻辑，传进来的参数就是缓存的最大容量，后面初始化函数定义了很多的属性，包含当前缓存的大小 cur_idx，cache 字典，双向链表的虚拟头节点和尾节点 \_\_hard\_head，\_\_hard\_tail，之后构建头节点和尾节点的弱引用代理 \_\_head，\_\_tail 然后构建一下链表，定义一下操作栈。

```python
class LRU:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cur_idx = 0
        self.cache = {}
        self.__hard_head = LinkNode()
        self.__hard_tail = LinkNode()
        self.__head = _proxy(self.__hard_head)
        self.__head.key = '___head___'
        self.__tail = _proxy(self.__hard_tail)
        self.__tail.key = '___tail___'
        self.__head.next = self.__tail
        self.__tail.prev = self.__head
        self.__head.prev = self.__tail.next = None
        self.op_stack = []
    ...
```

- get 函数，获取缓存中的值，键存在的话，将其移动到链表的头部，不存在的话，将其放到缓存里面，发现当前的 LRU 还有空间的话，将当前的键值对放到头节点上面，满了的话，将当前的尾节点删掉，将其放到头部，将 cache 里面老的键值对删掉。

```python
    ...
    def get(self, key):
        if key in self.cache:
            cur_node = self.cache[key]
            value = cur_node.value
            prev_node, next_node = cur_node.prev, cur_node.next
            next_node.prev = prev_node
            prev_node.next = next_node
            cur_node.next = self.__head.next
            self.__head.next.prev = cur_node
            cur_node.prev = self.__head
            self.__head.next = cur_node
            return value
        else:
            new_node = LinkNode()
            new_node.key = key
            if self.cur_idx < self.capacity:
                r = self.cur_idx
                self.cache[key] = new_node
                new_node.value = r
                self.cur_idx += 1
                new_node.next = self.__head.next
                self.__head.next.prev = new_node
                new_node.prev = self.__head
                self.__head.next = new_node
                return r
            else:
                free_node = self.__tail.prev
                free_node.prev.next = self.__tail
                self.__tail.prev = free_node.prev
                old_key = free_node.key
                r = free_node.value
                free_node.prev = free_node.next = None
                self.cache.pop(old_key)
                new_node.next = self.__head.next
                self.__head.next.prev = new_node
                self.__head.next = new_node
                new_node.prev = self.__head
                new_node.value = r
                self.cache[key] = new_node
                return r
    ...
```

- \_\_iter\_\_ 函数，构建我们 DCP 里面的迭代器，方便进行迭代 DCP 里面的内容，yield 作用类似于 return 但是他是惰性的返回，每次用 next() 调用只返回第一个，下次返回第二个，以此类推，节省内存，适合大量数据的处理。

```python
    ...
    def __iter__(self):
        cur = self.__head.next
        while cur.key != self.__tail.key:
            yield cur.key, cur.value
            cur = cur.next
    ...
```

- state_dict 函数，得到当前的 DCP 里面的内容，输出键值对关系。

```python
    ...
    def state_dict(self):
        cur = self.__head.next
        kvs = []
        while cur.key != self.__tail.key:
            kvs.append((cur.key, cur.value))
            cur = cur.next
        return kvs
    ...
```

- restore 函数，恢复函数，将一组键值对恢复到缓存里面，首先检查一下待恢复数据的 DCP 占用是多少，其次也要保证当前的 DCP为空，之后将这些键值对添加到 DCP 里面，当前的键值对必须存在缓存里面，否则不能恢复。

```python
    ...
    def restore(self, kvs):
        assert len(kvs) <= self.capacity
        assert self.cur_idx == 0
        cur_node = self.__head
        for kv in kvs:
            assert kv[0] not in self.cache
            new_node = LinkNode()
            new_node.key = kv[0]
            new_node.value = kv[1]
            self.cache[kv[0]] = new_node
            cur_node.next = new_node
            new_node.prev = cur_node
            cur_node = new_node
            self.cur_idx += 1
        cur_node.next = self.__tail
        self.__tail.prev = cur_node
    ...
```

- clear 函数，清空我们的链表 DCP。

```python
    ...
    def clear(self):
        cur_node = self.__head.next
        while cur_node != self.__tail:
            next_node = cur_node.next
            cur_node.prev = None
            cur_node.next = None
            cur_node = next_node
        self.cache.clear()
        self.__head.next = self.__tail
        self.__tail.prev = self.__head
    ...
```

- \_\_contains\_\_、view、keys 函数。

```python
    ...
    def __contains__(self, key):
        return key in self.cache
    def view(self, key):
        if key not in self.cache:
            return -1
        else:
            return self.cache[key].value
    def keys(self):
        return self.cache.keys()
    ...
```

- try\_get 函数，类似 get 函数，主要的区别是会将操作记录记录在 op_stack 里面，方便我们回滚。

```python
    ...
    def try_get(self, key):
        if key in self.cache:
            cur_node = self.cache[key]
            value = cur_node.value
            prev_node, next_node = cur_node.prev, cur_node.next
            op_records = Op('Get', prev_node, next_node)
            self.op_stack.append(op_records)
            next_node.prev = prev_node
            prev_node.next = next_node
            cur_node.next = self.__head.next
            self.__head.next.prev = cur_node
            cur_node.prev = self.__head
            self.__head.next = cur_node
            return value
        else:
            new_node = LinkNode()
            new_node.key = key
            if self.cur_idx < self.capacity:
                r = self.cur_idx
                self.cache[key] = new_node
                new_node.value = r
                self.cur_idx += 1
                new_node.next = self.__head.next
                self.__head.next.prev = new_node
                new_node.prev = self.__head
                self.__head.next = new_node
                op_records = Op('Add', new_node.prev, new_node.next, new_node)
                self.op_stack.append(op_records)
                return r
            else:
                free_node = self.__tail.prev
                free_node.prev.next = self.__tail
                self.__tail.prev = free_node.prev
                old_key = free_node.key
                r = free_node.value
                op_records = Op(
                    'Overflow', free_node.prev, free_node.next, free_node, key=old_key, value=free_node.value
                )
                self.op_stack.append(op_records)
                free_node.prev = free_node.next = None
                self.cache.pop(old_key)
                new_node.next = self.__head.next
                self.__head.next.prev = new_node
                self.__head.next = new_node
                new_node.prev = self.__head
                new_node.value = r
                self.cache[key] = new_node
                return r
    ...
```

- rollback_one_step、rollback_steps 函数，回滚和多次回滚，回滚操作，当前的 op_stack 里面有数据的话，我们需要对里面的操作进行一次回滚，当回滚 Add 操作的时候相当于删除添加的节点，回滚 Overflow 的时候，这个比较复杂，本来就是将尾巴的节点删掉，将当前的节点加到头上，所以回滚就是反向的操作，先就是移除新的节点，然后添加删除的旧的节点，Get 节点因为是把节点移动到了头节点，现在要做的就是吧头节点插回去。 多次回滚的话就是多调用几遍单次回滚。

```python
    ...
    def rollback_one_step(self, ):
        if len(self.op_stack) > 0:
            last_op = self.op_stack.pop()
            if last_op.op_type == 'Add':
                removed_node = last_op.removed_node
                prev_node, next_node = last_op.prev_node, last_op.next_node
                prev_node.next = next_node
                next_node.prev = prev_node
                removed_node.prev = removed_node.next = None
                self.cache.pop(removed_node.key)
                self.cur_idx -= 1
            elif last_op.op_type == 'Overflow':
                new_node = self.__head.next
                self.__head.next = new_node.next
                new_node.next.prev = self.__head
                new_node.next = new_node.prev = None
                self.cache.pop(new_node.key)

                removed_node = last_op.removed_node
                prev_node = last_op.prev_node
                next_node = last_op.next_node
                prev_node.next = removed_node
                removed_node.prev = prev_node
                removed_node.next = next_node
                next_node.prev = removed_node
                key = last_op.kv[0]
                removed_node.key = key
                removed_node.value = last_op.kv[1]
                assert key not in self.cache
                self.cache[key] = removed_node
            else:
                assert last_op.op_type == 'Get'
                cur_node = self.__head.next
                self.__head.next = cur_node.next
                cur_node.next.prev = self.__head
                last_op.prev_node.next = cur_node
                cur_node.prev = last_op.prev_node
                cur_node.next = last_op.next_node
                last_op.next_node.prev = cur_node
    def rollback_steps(self, steps):
        max_steps = min(steps, len(self.op_stack))
        for _ in range(max_steps):
            self.rollback_one_step()
```

## 4 >> 构建步骤

你现在已经完成了 LRU 策略的设计，检查没有问题的话，还剩下：

- 005 >> [FFC-DCP动态池](https://github.com/sqnkkang/Very-Large-Scale-Face-Recognition/blob/master/recognition/ffc_dcp.md)
- 006 >> [训练](https://github.com/sqnkkang/Very-Large-Scale-Face-Recognition/blob/master/recognition/train.md)
- 007 >> [测试](https://github.com/sqnkkang/Very-Large-Scale-Face-Recognition/blob/master/recognition/test.md)

## 5 >> 致谢

本文受 [Build-Your-Own-Face-Model](https://github.com/siriusdemon/Build-Your-Own-Face-Model/) 与 [FFC](https://github.com/tiandunx/FFC/) 的启发，主要用作作者学习使用。
