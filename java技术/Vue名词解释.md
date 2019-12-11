# Vue名词解释

text interpolation 将文本嵌入到{{}}中

directive v- 系列指代它是Vue提供的特别属性。它可以为其渲染的DOM提供特殊的反应式属性。

v-model 在输入和网页状态直接进行绑定。

v-on：operation，当用户对当前元素执行某操作时调用的函数。

v-bind:props = "item", v-bind:key = "item.id"

v-html = "rawhtml", 慎用，可能会导致XSS攻击。

instance lifecucle hooks: created, mounted, updated, destroyed. 

Dynamic Arguments：动态参数，v-bind:[attr],这里的attr可以是一个js变量

Modifiers：修饰器的作用是directive应该以某种特殊方式绑定

computed: 不应在原始数据修改，应先copy再修改并返回新表示，当想要修改数据时使用methods。只有当computed依赖的值改变了，它的值才会变，而method则是每次调用都会更新一遍值。

Computed Setter: set: function(newValue){

对computed的值进行相应的修改，绝大多数的情况下修改其组成即可，get方法的返回值会自动根据新的值更新。

}

**a gentle example:**

![1575973460653](C:\Users\YOGA710\AppData\Roaming\Typora\typora-user-images\1575973460653.png)

 ![The Vue Instance Lifecycle](https://vuejs.org/images/lifecycle.png) 