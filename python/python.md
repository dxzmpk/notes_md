# Python 编程小记

## 1. 30个必要的知识点和小技能

### Tips#1. 两个数的交换

```python
x, y = 10, 20
print(x, y)
 
x, y = y, x
print(x, y)
 
#1 (10, 20)
#2 (20, 10)
```

这里首先由右侧的赋值运算符产生一个新元组（20，10）,然后左侧直接将其解释为(unpack)名称x,y。然后临时的元组就会被垃圾清理。

### Tips#2. 比较符的链式法则

```python
n = 10
result = 1 < n < 20
print(result)

# True

result = 1 > n <= 9
print(result)

# False
```

### Tips#3. 使用三元运算符实现条件赋值

Ternary operators are a shortcut for an if-else statement and also known as conditional operators.

```
[on_true] if [expression] else [on_false]
```

Here are a few examples which you can use to make your code 压缩而准确.

The below statement is doing the same what it is meant to i.e. “**assign 10 to x if y is 9, otherwise assign 20 to x**“. We can though extend the chaining of operators if required.

```
x = 10 if (y == 9) else 20
```

Likewise, we can do the same for class objects.

```
x = (classA if y == 1 else classB)(param1, param2)
```

在上例中，classA 和 classB是两个类对象。两个类构造方法中将有一个被调用。

Below is one more example with a no. of conditions joining to evaluate the smallest number.

```python
def small(a, b, c):
	return a if a <= b and a <= c else (b if b <= a and b <= c else c)
print(small(1, 0, 1))
print(small(1, 2, 2))
print(small(2, 2, 3))
print(small(5, 4, 3))

#Output
#0 #1 #2 #3
```

We can even use a ternary operator with the list comprehension.



```
[m**2 if m > 10 else m**4 for m in range(50)]

#=> [0, 1, 16, 81, 256, 625, 1296, 2401, 4096, 6561, 10000, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400, 441, 484, 529, 576, 625, 676, 729, 784, 841, 900, 961, 1024, 1089, 1156, 1225, 1296, 1369, 1444, 1521, 1600, 1681, 1764, 1849, 1936, 2025, 2116, 2209, 2304, 2401]
```

### Tips#4. 多行字符串

The basic approach is to use backslashes which derive itself from C language.

```
multiStr = "select * from multi_row \
where row_id < 5"
print(multiStr)

# select * from multi_row where row_id < 5
```

One more trick is to use the triple-quotes.

```
multiStr = """select * from multi_row 
where row_id < 5"""
print(multiStr)

#select * from multi_row 
#where row_id < 5
```

The common issue with the above methods is the lack of proper indentation. If we try to indent, it’ll insert whitespaces in the string.

So the final solution is to split the string into multi lines and enclose the entire string in parenthesis.

```
multiStr= ("select * from multi_row "
"where row_id < 5 "
"order by age") 
print(multiStr)

#select * from multi_row where row_id < 5 order by age
```

### Tips#5. 将列表中的元素存储到多个变量中

We can use a list to initialize a no. of variables. 当解压缩列表时，变量的数目应等于列表中元素的个数。

```
testList = [1,2,3]
x, y, z = testList

print(x, y, z)

#-> 1 2 3
```

### Tips#6. 打印导入模块的路径

If you want to know the absolute location of modules imported in your code, then use the below trick.

```
import threading 
import socket

print(threading)
print(socket)

#1- <module 'threading' from '/usr/lib/python2.7/threading.py'>
#2- <module 'socket' from '/usr/lib/python2.7/socket.py'>
```

### Tips#7. 在交互模式中使用 “_” 运算符.

当我们测试变量或者调用函数时，结果会自动存储到`_`运算符中

```
>>> 2 + 1
3
>>> _
3
>>> print _
3
```

The “_” references to the output of the last executed expression.

### Tips#8. 字典/集合 Comprehensions. 

 Comprehensions指的是从旧列表产生新列表的方法。

```
testDict = {i: i * i for i in xrange(10)} 
testSet = {i * 2 for i in xrange(10)}

print(testSet)
print(testDict)

#set([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
#{0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36, 7: 49, 8: 64, 9: 81}
```

**Note-** There is only a difference of <:> in the two statements. Also, to run the above code in Python3, replace <xrange> with <range>.

### Tips#9. 调试脚本（使用断点）

We can set breakpoints in our Python script with the help of the <pdb> module. Please follow the below example.

```
import pdb
pdb.set_trace()
```

We can specify <pdb.set_trace()> anywhere in the script and set a breakpoint there. It’s extremely convenient.

python3.7可以使用`breakpoint()`,相当于上面两行代码

### Tips#10. 使用python进行文件分享

Python allows running an HTTP server which you can use to share files from the server root directory. Below are the commands to start the server.

#### # Python 2

```
python -m SimpleHTTPServer
```

#### # Python 3

```
python3 -m http.server
```

Above commands would start a server on the default port i.e. 8000. You can also use a custom port by passing it as the last argument to the above commands.

### Tips#11. 观察python对象的详细信息

We can inspect objects in Python by calling the dir() method. Here is a simple example.

```
test = [1, 3, 5, 7]
print( dir(test) )
['__add__', '__class__', '__contains__', '__delattr__', '__delitem__', '__delslice__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__getslice__', '__gt__', '__hash__', '__iadd__', '__imul__', '__init__', '__iter__', '__le__', '__len__', '__lt__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__rmul__', '__setattr__', '__setitem__', '__setslice__', '__sizeof__', '__str__', '__subclasshook__', 'append', 'count', 'extend', 'index', 'insert', 'pop', 'remove', 'reverse', 'sort']
```

### Tips#12. 简化if

To verify multiple values, we can do in the following manner.

```
if m in [1,3,5,7]:
```

instead of:

```
if m==1 or m==3 or m==5 or m==7:
```

Alternatively, we can use ‘{1,3,5,7}’ instead of ‘[1,3,5,7]’ for ‘in’ operator **because ‘set’ can access each element by O(1).** 

### Tips#13. 运行时检测python版本

Sometimes we may not want to execute our program if the Python engine currently running is less than the supported version. To achieve this, you can use the below coding snippet. It also prints the currently used Python version in a readable format.

```
import sys

#Detect the Python version currently in use.
if not hasattr(sys, "hexversion") or sys.hexversion != 50660080:
    print("Sorry, you aren't running on Python 3.5\n")
    print("Please upgrade to 3.5.\n")
    sys.exit(1)
    
#Print Python version in a readable format.
print("Current Python version: ", sys.version)
```

Alternatively, you can use`sys.version_info >= (3, 5)` to replace`sys.hexversion!= 50660080` in the above code. It was a suggestion from one of the informed reader.

Output when running on Python 2.7.

```
Python 2.7.10 (default, Jul 14 2015, 19:46:27)
[GCC 4.8.2] on linux
   
Sorry, you aren't running on Python 3.5

Please upgrade to 3.5.
```

Output when running on Python 3.5.

```
Python 3.5.1 (default, Dec 2015, 13:05:11)
[GCC 4.8.2] on linux
   
Current Python version:  3.5.2 (default, Aug 22 2016, 21:11:05) 
[GCC 5.3.0]
```

### Tips#14. 将多个字符串连接在一起

If you want to concatenate all the tokens available in a list, then see the below example.

```
>>> test = ['I', 'Like', 'Python', 'automation']
```

Now, let’s create a single string from the elements in the list given above.

```
>>> print ''.join(test)
```

### Tips#15. 四种反转字符串/列表的方式

#### # Reverse The List Itself.

```
testList = [1, 3, 5]
testList.reverse()
print(testList)

#-> [5, 3, 1]
```

#### # Reverse While Iterating In A Loop.

```
for element in reversed([1,3,5]): print(element)

#1-> 5
#2-> 3
#3-> 1
```

#### # Reverse A String In Line.

```
"Test Python"[::-1]
```

This gives the output as ”nohtyP tseT”

#### # Reverse A List Using Slicing.

```
[1, 3, 5][::-1]
```

The above command will give the output as [5, 3, 1].

### Tips#16. 使用枚举（Enumeration）.

With enumerators, it’s easy to find an index while you’re inside a loop.

```
testlist = [10, 20, 30]
for i, value in enumerate(testlist):
	print(i, ': ', value)

#1-> 0 : 10
#2-> 1 : 20
#3-> 2 : 30
```

### Tips#17. 使用枚举变量（Enums)

We can use the following approach to create enum definitions.

```
class Shapes:
	Circle, Square, Triangle, Quadrangle = range(4)

print(Shapes.Circle)
print(Shapes.Square)
print(Shapes.Triangle)
print(Shapes.Quadrangle)

#1-> 0
#2-> 1
#3-> 2
#4-> 3
```

### Tips#18. 函数返回多个值

Not many programming languages support this feature. However, functions in Python do return multiple values.

Please refer the below example to see it working.

```
# function returning multiple values.
def x():
	return 1, 2, 3, 4

# Calling the above function.
a, b, c, d = x()

print(a, b, c, d)

#-> 1 2 3 4
```

### Tips#19. 使用冲击运算符取得参数

The splat operator offers an artistic way to unpack arguments lists. Please refer the below example for clarity.

```
def test(x, y, z):
	print(x, y, z)

testDict = {'x': 1, 'y': 2, 'z': 3} 
testList = [10, 20, 30]

test(*testDict)
test(**testDict)
test(*testList)

#1-> x y z
#2-> 1 2 3
#3-> 10 20 30
```

### Tips#20. 使用字典存储表达式

We can make a dictionary store expressions.

```
stdcalc = {
	'sum': lambda x, y: x + y,
	'subtract': lambda x, y: x - y
}

print(stdcalc['sum'](9,3))
print(stdcalc['subtract'](9,3))

#1-> 12
#2-> 6
```

### Tips#21. 在一行中计算数字的阶乘

#### Python 2.X.

```
result = (lambda k: reduce(int.__mul__, range(1,k+1),1))(3)
print(result)
#-> 6
```

#### Python 3.X.

```
import functools
result = (lambda k: functools.reduce(int.__mul__, range(1,k+1),1))(3)
print(result)

#-> 6
```

### Tips#22. 找到列表中出现次数最多的值

```
test = [1,2,3,4,2,2,3,1,4,4,4]
print(max(set(test), key=test.count))

#-> 4
```

```
max(iterable, *iterables[, key, default])
key (Optional) - key function where the iterables are passed and comparison is performed based on its return value
default (Optional) - default value if the given iterable is empty
```

### Tips#23. 重置递归限制。

Python restricts recursion limit to 1000. We can though reset its value.

```
import sys

x=1001
print(sys.getrecursionlimit())

sys.setrecursionlimit(x)
print(sys.getrecursionlimit())

#1-> 1000
#2-> 1001
```

Please apply the above trick only if you need it.

### Tips#24. 查看对象的内存使用

In Python 2.7, a 32-bit integer consumes 24-bytes whereas it utilizes 28-bytes in Python 3.5. To verify the memory usage, we can call the <getsizeof> method.

#### In Python 2.7.

```
import sys
x=1
print(sys.getsizeof(x))

#-> 24
```

#### In Python 3.5.

```
import sys
x=1
print(sys.getsizeof(x))

#-> 28
```

### Tips#25. 使用__slots__减少内存开销

Have you ever observed your Python application consuming a lot of resources especially memory? Here is one trick which uses <__slots__> class variable to reduce memory overhead to some extent.

```python
import sys
class FileSystem(object):

	def __init__(self, files, folders, devices):
		self.files = files
		self.folders = folders
		self.devices = devices

print(sys.getsizeof( FileSystem ))

class FileSystem1(object):

	__slots__ = ['files', 'folders', 'devices']
	
	def __init__(self, files, folders, devices):
		self.files = files
		self.folders = folders
		self.devices = devices

print(sys.getsizeof( FileSystem1 ))

#In Python 3.5
#1-> 1016
#2-> 888
```

Clearly, you can see from the results that there are savings in memory usage. But you should use __slots__ when the memory overhead of a class is unnecessarily large. Do it only after profiling the application. Otherwise, you’ll make the code difficult to change and with no real benefit.

### Tips#26. Lambda模仿打印功能。

```
import sys
lprint=lambda *args:sys.stdout.write(" ".join(map(str,args)))
lprint("python", "tips",1000,1001)

#-> python tips 1000 1001
```

```python
map(function, iterable, ...)
function - map() passes each item of the iterable to this function.
iterable iterable which is to be mapped
```

### Tips#27. 从两个相关序列中产生新字典

```
t1 = (1, 2, 3)
t2 = (10, 20, 30)

print(dict (zip(t1,t2)))

#-> {1: 10, 2: 20, 3: 30}
```

### Tips#28. 在一行代码中查找字符串的多个可能出现的序列

```
print("http://www.google.com".startswith(("http://", "https://")))
print("http://www.google.co.uk".endswith((".com", ".co.uk")))

#1-> True
#2-> True
```

### Tips#29. 将多维列表展开，不使用任何循环。

```
import itertools
test = [[-1, -2], [30, 40], [25, 35]]
print(list(itertools.chain.from_iterable(test)))

#-> [-1, -2, 30, 40, 25, 35]
```

If you have an input list with nested lists or tuples as elements, then use the below trick. However, the limitation here is that it’s using a for Loop.

```
def unifylist(l_input, l_target):
    for it in l_input:
        if isinstance(it, list):
            unifylist(it, l_target)
        elif isinstance(it, tuple):
            unifylist(list(it), l_target)
        else:
            l_target.append(it)
    return l_target

test =  [[-1, -2], [1,2,3, [4,(5,[6,7])]], (30, 40), [25, 35]]

print(unifylist(test,[]))

#Output => [-1, -2, 1, 2, 3, 4, 5, 6, 7, 30, 40, 25, 35]
```

Another simpler method to unify the list containing lists and tuples is by using the Python’s <**more_itertools**> package. It doesn’t require looping. Just do a <**pip install more_itertools**>, if not already have it.

```
import more_itertools

test = [[-1, -2], [1, 2, 3, [4, (5, [6, 7])]], (30, 40), [25, 35]]

print(list(more_itertools.collapse(test)))

#Output=> [-1, -2, 1, 2, 3, 4, 5, 6, 7, 30, 40, 25, 35]
```

### Tips#30. 在Python中实现真假判断Switch-Case语句。

Here is the code that uses a dictionary to imitate a switch-case construct.

```
def xswitch(x): 
	return xswitch._system_dict.get(x, None) 

xswitch._system_dict = {'files': 10, 'folders': 5, 'devices': 2}

print(xswitch('default'))
print(xswitch('devices'))

#1-> None
#2-> 2
```

### Tips#31. 使用Counter查看两个字符串是否含有同样的字符


Complete the above method to find if two words are anagrams.

```python
from collections import Counter
def is_anagram(str1, str2):
     return Counter(str1) == Counter(str2)
>>> is_anagram('abcd','dbca')
True
>>> is_anagram('abcd','dbaa')
False
```

### Tips#32. 一行代码实现将字符串转换为整数数组

```python
>>> result = map(lambda x:int(x) ,raw_input().split())
1 2 3 4
>>> result
[1, 2, 3, 4]
```



## 2. range 和 xrange的区别

Before we get started, let's talk about what makes `xrange` and `range` different.

For the most part, `xrange` and `range` are the exact same in terms of functionality. They both provide a way to generate a list of integers for you to use, however you please. **The only difference is that `range` returns a Python `list` object and `xrange` returns an `xrange` object.**

What does that mean? Good question! It means that `xrange` doesn't actually generate a static list at run-time like `range` does. It creates the values as you need them with a special technique called *yielding*. This technique is used with a type of object known as *generators*. If you want to read more in depth about generators and the yield keyword, be sure to checkout the article [Python generators and the yield keyword](https://www.pythoncentral.io/python-generators-and-yield-keyword/).

Okay, now what does *that* mean? Another good question. *That* means that if you have a really gigantic range you'd like to generate a list for, say one billion, `xrange` is the function to use. This is especially true if you have a really memory sensitive system such as a cell phone that you are working with, as `range` will use as much memory as it can to create your array of integers, which can result in a `MemoryError` and crash your program. It's a memory hungry beast.

如果需要多次遍历list,那么最好使用`range`，因为它是静态的，所有对象都已经创建好了，而`xrange`需要在你每次调用它时新创建变量。这会浪费大量时间。

在python3中，xrange被range的功能替代了

```python
import sys
if sys.version_info < (3,):
    range = xrange
```

## 3.python精致画图技巧

1. 少即是多

    *完美的实现不是在没有其他东西可添加时，而是在没有其他东西可以拿走时实现的。* 

2. 颜色很重要

   使用Tableau的默认配色方案

   使用众所周知的产生精美图的软件中已建立的默认配色方案。Tableau [有一套](https://tableaufriction.blogspot.ro/2012/11/finally-you-can-use-tableau-data-colors.html)出色[的配色方案](https://tableaufriction.blogspot.ro/2012/11/finally-you-can-use-tableau-data-colors.html)，从灰度到有色到对色盲友好。这使我想到了下一个……

   许多图形设计师完全忘记了[色盲](https://en.wikipedia.org/wiki/Color_blindness)，[色盲](https://en.wikipedia.org/wiki/Color_blindness)会影响超过5％的图形查看者。例如，对于使用红绿色盲的人来说，使用红色和绿色区分两类数据的图将是[完全](http://99designs.com/designer-blog/2013/04/17/designers-need-to-understand-color-blindness/)无法[理解](http://99designs.com/designer-blog/2013/04/17/designers-need-to-understand-color-blindness/)的。尽可能使用对色盲友好的配色方案，例如[Tableau的“ Color Blind 10”。](https://tableaufriction.blogspot.ro/2012/11/finally-you-can-use-tableau-data-colors.html)

3. 其他库

   ### More Python plotting libraries

   In this tutorial, I focused on making data visualizations with only Python’s basic matplotlib library. If you don’t feel like tweaking the plots yourself and want the library to produce better-looking plots on its own, check out the following libraries.

   - [Seaborn](https://seaborn.pydata.org/) for statistical charts
   - [ggplot2 for Python](http://ggplot.yhathq.com/)
   - [prettyplotlib](https://olgabot.github.io/prettyplotlib/)
   - [Bokeh](https://bokeh.pydata.org/) for interactive charts

## 4. python编码规范 PEP 8

遵守pep 8意味着：

1. 你的变量命名符合规范
2. 添加了足够多的空格，逻辑顺序易读
3. 注释足够

几个窍门：

1. 对于算数表达式，采用多行可以更清楚地看到哪个操作数对应哪一个运算符

   ```python
   # Recommended
   total = (first_variable
            + second_variable
            - third_variable)
   ```

2. 括号的关闭要和前文对齐

   ```
   list_of_numbers = [
       1, 2, 3,
       4, 5, 6,
       7, 8, 9
       ]
   ```

   ```
   list_of_numbers = [
       1, 2, 3,
       4, 5, 6,
       7, 8, 9
   ]
   ```

### Comments

> “If the implementation is hard to explain, it’s a bad idea.”
>
> — *The Zen of Python*

1. 对于多行代码，写注释解释很有必要

   ```python
   def quadratic(a, b, c, x):
       # Calculate the solution to a quadratic equation using the quadratic
       # formula.
       #
       # There are always two solutions to a quadratic equation, x_1 and x_2.
       x_1 = (- b+(b**2-4*a*c)**(1/2)) / (2*a)
       x_2 = (- b-(b**2-4*a*c)**(1/2)) / (2*a)
       return x_1, x_2
   ```

2. doc strings要写在所有的公共模块，功能，类，方法上

   ```python
   def quadratic(a, b, c, x):
       """Solve quadratic equation via the quadratic formula.
   
       A quadratic equation has the following form:
       ax**2 + bx + c = 0
   
       There always two solutions to a quadratic equation: x_1 & x_2.
       """
       x_1 = (- b+(b**2-4*a*c)**(1/2)) / (2*a)
       x_2 = (- b-(b**2-4*a*c)**(1/2)) / (2*a)
   
       return x_1, x_2
   ```

   ```python
   def quadratic(a, b, c, x):
       """Use the quadratic formula"""
       x_1 = (- b+(b**2-4*a*c)**(1/2)) / (2*a)
       x_2 = (- b-(b**2-4*a*c)**(1/2)) / (2*a)
   
       return x_1, x_2
   ```

3. 空格环绕运算符--除了用在函数默认参数上

   ```
   # Recommended
   def function(default_parameter=5):
       # ...
   
   
   # Not recommended
   def function(default_parameter = 5):
       # ...
   ```

   **当一句代码中有多个运算符时，应该只在最低级的运算符周围添加空格。**

   ```python
   # Recommended
   y = x**2 + 5
   z = (x+y) * (x-y)
   
   # Not Recommended
   y = x ** 2 + 5
   z = (x + y) * (x - y)
   ```

   ```python
   # Recommended
   if x>5 and x%2==0:
       print('x is larger than 5 and divisible by 2!')
   ```

   在切片操作时，冒号也作为二元运算符出现。

   ```python
   list[3:4]
   
   # Treat the colon as the operator with lowest priority
   list[x+1 : x+2]
   
   # In an extended slice, both colons must be
   # surrounded by the same amount of whitespace
   list[3:4:5]
   list[x+1 : x+2 : x+3]
   
   # The space is omitted if a slice parameter is omitted
   list[x+1 : x+2 :]
   ```

### Programming Recommendations

> “Simple is better than complex.”
>
> — *The Zen of Python*

1. 不要用布尔值和True或者False作比较, less code and simpler

   ```python
   # Recommended
   if my_bool:
       return '6 is bigger than 5'
   ```

2. 空列表默认布尔值为`False`

   ```python
   # Recommended
   my_list = []
   if not my_list:
       print('List is empty!')
   ```

3. 检查输入参数是否为空时，使用：

   ```python
   # Recommended
   if arg is not None:
       # Do something with arg...
       
   # Not Recommended
   if arg:
       # Do something with arg...
   ```

   The mistake being made here is assuming that `not None` and truthy are equivalent. You could have set `arg = []`. As we saw above, empty lists are evaluated as falsy in Python. So, even though the argument `arg` has been assigned, the condition is not met, and so the code in the body of the `if` statement will not be executed. 

## 5. 学习python的小技巧

1. 使用`dir()`，`type`，`help`
2. 问GOOD问题
   - **G**: Give context on what you are trying to do, clearly describing the problem.
   - **O**: Outline the things you have already tried to fix the issue.
   - **O**: Offer your best guess as to what the problem might be. This helps the person who is helping you to not only know what you are thinking, but also know that you have done some thinking on your own.
   - **D**: Demo what is happening. Include the code, a traceback error message, and an explanation of the steps you executed that resulted in the error. This way, the person helping does not have to try to recreate the issue.

## 6. 如何让python代码[跑得更快](http://earthpy.org/speed.html)

numpy, scipy和pandas大部分都是由C编写的，所以它们跑起来比较快。同时，使用向量化运算也会加快运算。同时要避免循环，当非用不可的时候，应该选择以下几种方式让循环跑得更快。

1. 多处理器

   ```python
   #look at how many processors your computer have
   import multiprocessing
   
   multiprocessing.cpu_count()
   
   pool = multiprocessing.Pool(processes = 2)
   
   r = pool.map(function, parameters_list)
   pool.close()
   ```

2. 使用Cython

   ```python
   %%cython
   def useless_cython(year):
       from netCDF4 import Dataset
       f = Dataset('air.sig995.'+year+'.nc')
       a = f.variables['air'][:]
       a_cum = 0
       for i in range(a.shape[0]):
           for j in range(a.shape[1]):
               for n in range(a.shape[2]):
                   a_cum = a_cum+a[i,j,n]
                   
       a_cum.tofile(year+'.bin')
       print(year)
       return a_cum
   ```

   But the true power of cython revealed only when you provide types of your variables. You have to use `cdef` keyword in the function definition to do so. There are also couple other modifications to the function

   ```python
   %%cython
   import numpy as np
   
   def useless_cython(year):
       
       # define types of variables
       cdef int i, j, n
       cdef double a_cum
       
       from netCDF4 import Dataset
       f = Dataset('air.sig995.'+year+'.nc')
       a = f.variables['air'][:]
       
       a_cum = 0.
       for i in range(a.shape[0]):
           for j in range(a.shape[1]):
               for n in range(a.shape[2]):
                   #here we have to convert numpy value to simple float
                   a_cum = a_cum+float(a[i,j,n])
       
       # since a_cum is not numpy variable anymore,
       # we introduce new variable d in order to save
       # data to the file easily
       d = np.array(a_cum)
       d.tofile(year+'.bin')
       print(year)
       return d
   ```

3. Numba

   Numba is an just-in-time specializing compiler which compiles annotated Python and NumPy code to LLVM (through decorators). The easiest way to install it is to use Anaconda distribution.

   ```
   from numba import jit, autojit
   ```

   We now have to split our function in two (that would be a good idea from the beggining). One is just number crunching part, and another responsible for IO. The only thing that we have to do afterwards is to put `jit` decorator in front of the first function.

   ```python
   @autojit
   # @jit('f8(f4[:,:,:])') provide type means faster
   def calc_sum(a):
       a_cum = 0.
       for i in range(a.shape[0]):
           for j in range(a.shape[1]):
               for n in range(a.shape[2]):
                   a_cum = a_cum+a[i,j,n]
       return a_cum
   
   def useless_numba(year):
       #from netCDF4 import Dataset
       f = Dataset('air.sig995.'+year+'.nc')
       a = f.variables['air'][:]
       a_cum = calc_sum(a)
       
       d = np.array(a_cum)
       d.tofile(year+'.bin')
       print(year)
       return d
   ```

4. 使用本地numpy方法 -- 在处理求和问题时效果最佳

   ```python
   import numpy as np
   def calc_sum(a):
       a = np.float64(a)
       return a.sum()
   
   def useless_good(year):
       from netCDF4 import Dataset
       f = Dataset('air.sig995.'+year+'.nc')
       a = f.variables['air'][:]
       a_cum = calc_sum(a)
       
       d = np.array(a_cum)
       d.tofile(year+'.bin')
       print(year)
       return d
   ```

   



## 7.使用python下载文件

```python
import requests

print('Beginning file download with requests')
url = 'https://timgsa.baidu.com/timg?image&amp;quality=80&amp;size=b9999_10000&amp;sec=1572927564810&amp;di=7886815cdb18cd4682340e337b6fd0a2&amp;imgtype=0&amp;src=http%3A%2F%2Fhbimg.b0.upaiyun.com%2F6b0018f329217b50b48d9478776a24c9825cbef7cd86-IrED2R_fw658'
url = 'http://www.nipic.com/show/7683069.html'
r = requests.get(url)

with open('./cat3.html', 'wb') as f:
    f.write(r.content)

#Retrieve HTTP meta-data

print(r.status_code)
print(r.headers['content-type'])
print(r.encoding)
```

## 8. Python中下划线的作用

![](https://www.runoob.com/wp-content/uploads/2018/10/v2-cbc5c6037101c7d33cf0acd9f00a8cfa_r.jpg)

## 9.python numpy[使用技巧](http://anie.me/numpy-tricks/)

之前写好的东西没有保存

### Trick 1: Collection1 == Collection2

```
>> X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
# Training labels (y_train) shape:  (50000,)
# to pick out correct y examples for one class (assuming that class is indexed at 5)
>> print(y_train == 5)
# array([False, False, False, ..., False, False, False], dtype=bool)
>> idxs = np.flatnonzero(y_train == 5)
```

`np.flatnonzero()` takes an array as input. Boolean array in Python has the property of `True` mapping to 1 and `False` mapping to 0, so `flatnonzero()` can easily pick out the index of those examples that are of class 5.

### Trick 2: Vectorization/Loop-elimination with `np.tile()`

Sometimes when you want to add two matrices together, and have to manually align/broadcast them, you can use np.tile(). This copy the current matrix and broadcast into the shape you want. Keep the lowest dimension as 1 so your matrix/vector stay the same.

### Trick 3: Using Array for Pair-wise Slicing

## Trick 3: Using Array for Pair-wise Slicing

Numpy array’s slicing often offers many pleasant surprises. This suprise comes from the context of SVM’s max() hinge-loss vectorization. SVM’s multi-class loss function requires wrong class scores to subtract correct class scores. When you dot product weight matrix and training matrix, you get a matrix shape of (num_train, num_classes). However, how do you get the score of correct classes out without looping (given y_labels of shape (num_train,))? At this situation, pair-wise selection could be helpful:

### Trick 4: Smart use of ‘:’ to extract the right shape

Sometimes you encounter a 3-dim array that is of shape (N, T, D), while your function requires a shape of (N, D). At a time like this, `reshape()` will do more harm than good, so you are left with one simple solution:

```
for t in xrange(T):
  x[:, t, :] = # ...
```

You can use it to extract values or assign values!

### Trick 5: Use Array as Slicing index

In previous posts, we already explored how Numpy array takes slicing of pairs (such as `x[range(x.shape[0]), y]`), however, Numpy can also take another array as slicing. Assume x is an index array of shape (N, T), each element index of x is in the range 0 <= idx < V, and we want to convert such index array into array with real weights, from a weight matrix w of shape (V, D), we can simply do:

```
N, T, V, D = 2, 4, 5, 3

x = np.asarray([[0, 3, 1, 2], [2, 1, 0, 3]])  # (N, T)
W = np.linspace(0, 1, num=V*D).reshape(V, D)  # (V, D)

# this is the only required line
out = W[x]  # (N, T, D)
```

Numpy uses the underlying value of x as index to extract values from W.

### Trick 6: Unfunc at

Numpy has a special group of “functions” called [unfunc](http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.ufunc.at.html). Turns out numpy.add, numpy.subtract and so on belong to this special group of functions.

Numpy has predefined methods for those functions such as `at`! This function performs the operation (let it be `add` or `subtract`) on a specific location (or locations).

`unfunc.at()` takes 3 arguments, first one is the matrix you intend to modify, second argument is the indices of the matrix you want to modify, third argument is the value you want to change (depending on what the `unfunc` function is).

A simple example is:

```
>>> a = np.array([1, 2, 3, 4])
>>> np.add.at(a, [0, 1, 2, 2], 1)
>>> print(a)
array([2, 3, 5, 4])
```

However, this example is too simple and almost useless. This kind of useless examples permeate the whole Numpy documentation. Let’s look at a more advanced example: use this trick to update a word embedding matrix!

```
# dout: Upstream gradients of shape (N, T, D)
# x: from example in Trick 5 (N, T), indices
# V: the length of total vocabulary
# D: weight matrix dimension
# task here is to build a dW to modify
# the original weight matrix

dW = np.zeros((V, D), dtype=dout.dtype)
np.add.at(dW, x, dout)
```

Notice that Numpy converts `x`, a matrix into individual indicies, and use it to assign values from dout to dW. `np.add.at()` flattened dout so the dimension becomes `(N*T, D)`. It will be checked against `x`’s dimension (N, T), and see if the product of x’s dimension and `N*T` will match. Only when this happens, you will assign the same amount of `D` arrays as you are instrcuted in `x`, to `dW`, which also happens to take `D` arrays.

## 10. python文档第一课

 Python是一种解释性语言，因为不需要编译和链接，因此可以在程序开发过程中节省大量时间。解释器可以交互使用，这使得在自下而上的程序开发过程中可以轻松地尝试语言的功能，编写一次性程序或测试功能。它也是一个方便的台式计算器。 

Python是可扩展的：如果您知道如何用C进行编程，则可以轻松地向解释器添加新的内置函数或模块，以最快的速度执行关键操作，或者将Python程序链接到可能仅以二进制形式可用的库（例如特定于供应商的图形库）。 一旦真正连接上，就可以将Python解释器链接到用C编写的应用程序中，并将其用作该应用程序的扩展或命令语言。

### 使用python[读入脚本的命令并进行相应操作]( https://www.tutorialspoint.com/python/python_command_line_arguments.htm )

```python
#!/usr/bin/python

import sys

print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)
```

```python
#!/usr/bin/python

import sys, getopt

def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print 'test.py -i <inputfile> -o <outputfile>'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print 'test.py -i <inputfile> -o <outputfile>'
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
   print 'Input file is "', inputfile
   print 'Output file is "', outputfile

if __name__ == "__main__":
   main(sys.argv[1:])
```

## 11. character encodings in python

字符串输出格式表示的规范：https://docs.python.org/3/library/string.html#formatspec

Here’s a more detailed look at each of these nine functions:

| Function  | Signature                                                    | Accepts                        | Return Type | Purpose                                                      |
| --------- | ------------------------------------------------------------ | ------------------------------ | ----------- | ------------------------------------------------------------ |
| `ascii()` | `ascii(obj)`                                                 | Varies                         | `str`       | ASCII only representation of an object, with non-ASCII characters escaped |
| `bin()`   | `bin(number)`                                                | `number: int`                  | `str`       | Binary representation of an integer, with the prefix `"0b"`  |
| `bytes()` | `bytes(iterable_of_ints)`  `bytes(s, enc[, errors])`  `bytes(bytes_or_buffer)`  `bytes([i])` | Varies                         | `bytes`     | Coerce (convert) the input to `bytes`, raw binary data       |
| `chr()`   | `chr(i)`                                                     | `i: int`  `i>=0`  `i<=1114111` | `str`       | Convert an integer code point to a single Unicode character  |
| `hex()`   | `hex(number)`                                                | `number: int`                  | `str`       | Hexadecimal representation of an integer, with the prefix `"0x"` |
| `int()`   | `int([x])`  `int(x, base=10)`                                | Varies                         | `int`       | Coerce (convert) the input to `int`                          |
| `oct()`   | `oct(number)`                                                | `number: int`                  | `str`       | Octal representation of an integer, with the prefix `"0o"`   |
| `ord()`   | `ord(c)`                                                     | `c: str`  `len(c) == 1`        | `int`       | Convert a single Unicode character to its integer code point |
| `str()`   | `str(object=’‘)`  `str(b[, enc[, errors]])`                  | Varies                         | `str`       | Coerce (convert) the input to `str`, text                    |