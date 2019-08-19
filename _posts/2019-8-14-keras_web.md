---
title: keras-web
data: 2019-8-14
categories: web
---

##### redis

https://www.jianshu.com/p/56999f2b8e3b

##### base64

Base64是一种用64个字符来表示任意二进制数据的方法。

图像的序列化与反序列化操作：

```python
def base64_encode_image(a):
	# base64 encode the input NumPy array
	return base64.b64encode(a).decode("utf-8")

def base64_decode_image(a, dtype, shape):
	# if this is Python 3, we need the extra step of encoding the
	# serialized NumPy string as a byte object
	if sys.version_info.major == 3:
		a = bytes(a, encoding="utf-8")

	# convert the string to a NumPy array using the supplied data
	# type and target shape
	a = np.frombuffer(base64.decodestring(a), dtype=dtype)
	a = a.reshape(shape)

	# return the decoded image
	return a
```

##### flask