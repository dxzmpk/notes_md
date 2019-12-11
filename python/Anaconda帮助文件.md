### Anaconda帮助文件

- [Managing environments](https://docs.anaconda.com/anaconda/navigator/tutorials/manage-environments/)
- [Managing packages](https://docs.anaconda.com/anaconda/navigator/tutorials/manage-packages/)
- [Managing channels](https://docs.anaconda.com/anaconda/navigator/tutorials/manage-channels/)

- [Using multiple versions of Python with Navigator](https://docs.anaconda.com/anaconda/navigator/tutorials/use-multiple-python-versions/) 
- [conda](https://conda.io/en/latest/)

## 使用国内镜像源

```
# 直接在后面添加 -i "https://pypi.doubanio.com/simple/"
# 例如安装 keras
pip install keras
# 变成
pip install keras -i "https://pypi.doubanio.com/simple/"
```

## pip与conda安装对比

1.pip安装不会安装所有的依赖项只会安装部分依赖项，而conda会安装全部；
2.pip与conda不会重复安装已经安装的依赖。
3.pip不会告诉你都安装了什么，conda会
4.pip安装的内容不会显示在anaconda navigation的环境中，conda会。
5.conda下载安装包位置xxxxx\anaconda\pkgs
6.conda list后面如果显示为说明是pip安装，不会显示在环境之中

## conda安装原理

1.先下载：下载到xxxxxx\anaconda\pkgs
2.再安装，安装到环境文件夹中xxxxxxx\anaconda\envs

[**使用pip安装**](https://blog.csdn.net/m0_37774696/article/details/84328843)

