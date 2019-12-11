# pytorch基本操作

# 1.计算梯度



```python
import torch

###############################################################
# Create a tensor and set ``requires_grad=True`` to track computation with it
x = torch.ones(2, 2, requires_grad=True)
print(x)

###############################################################
# Do a tensor operation:
y = x + 2
print(y)

###############################################################
# ``y`` was created as a result of an operation, so it has a ``grad_fn``.
print(y.grad_fn)

###############################################################
# Do more operations on ``y``
z = y * y * 3
out = z.mean()

print(z, out)

################################################################
# ``.requires_grad_( ... )`` changes an existing Tensor's ``requires_grad``
# flag in-place. The input flag defaults to ``False`` if not given.
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

###############################################################
# Gradients
# ---------
# Let's backprop now.
# Because ``out`` contains a single scalar, ``out.backward()`` is
# equivalent to ``out.backward(torch.tensor(1.))``.

out.backward()

###############################################################
# Print graddients d(out)/dx
#

print(x.grad)

###############################################################
# You should have got a matrix of ``4.5``. Let’s call the ``out``
# *Tensor* “:math:`o`”.
# We have that :math:`o = \frac{1}{4}\sum_i z_i`,
# :math:`z_i = 3(x_i+2)^2` and :math:`z_i\bigr\rvert_{x_i=1} = 27`.
# Therefore,
# :math:`\frac{\partial o}{\partial x_i} = \frac{3}{2}(x_i+2)`, hence
# :math:`\frac{\partial o}{\partial x_i}\bigr\rvert_{x_i=1} = \frac{9}{2} = 4.5`.
###############################################################
# Mathematically, if you have a vector valued function :math:`\vec{y}=f(\vec{x})`,
# then the gradient of :math:`\vec{y}` with respect to :math:`\vec{x}`
# is a Jacobian matrix:
#
# .. math::
#   J=\left(\begin{array}{ccc}
#    \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{1}}{\partial x_{n}}\\
#    \vdots & \ddots & \vdots\\
#    \frac{\partial y_{m}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
#    \end{array}\right)
#
# Generally speaking, ``torch.autograd`` is an engine for computing
# vector-Jacobian product. That is, given any vector
# :math:`v=\left(\begin{array}{cccc} v_{1} & v_{2} & \cdots & v_{m}\end{array}\right)^{T}`,
# compute the product :math:`v^{T}\cdot J`. If :math:`v` happens to be
# the gradient of a scalar function :math:`l=g\left(\vec{y}\right)`,
# that is,
# :math:`v=\left(\begin{array}{ccc}\frac{\partial l}{\partial y_{1}} & \cdots & \frac{\partial l}{\partial y_{m}}\end{array}\right)^{T}`,
# then by the chain rule, the vector-Jacobian product would be the
# gradient of :math:`l` with respect to :math:`\vec{x}`:
#
# .. math::
#   J^{T}\cdot v=\left(\begin{array}{ccc}
#    \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{1}}\\
#    \vdots & \ddots & \vdots\\
#    \frac{\partial y_{1}}{\partial x_{n}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
#    \end{array}\right)\left(\begin{array}{c}
#    \frac{\partial l}{\partial y_{1}}\\
#    \vdots\\
#    \frac{\partial l}{\partial y_{m}}
#    \end{array}\right)=\left(\begin{array}{c}
#    \frac{\partial l}{\partial x_{1}}\\
#    \vdots\\
#    \frac{\partial l}{\partial x_{n}}
#    \end{array}\right)
#
# (Note that :math:`v^{T}\cdot J` gives a row vector which can be
# treated as a column vector by taking :math:`J^{T}\cdot v`.)
#
# This characteristic of vector-Jacobian product makes it very
# convenient to feed external gradients into a model that has
# non-scalar output.

###############################################################
# Now let's take a look at an example of vector-Jacobian product:

x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

###############################################################
# Now in this case ``y`` is no longer a scalar. ``torch.autograd``
# could not compute the full Jacobian directly, but if we just
# want the vector-Jacobian product, simply pass the vector to
# ``backward`` as argument:
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)

###############################################################
# You can also stop autograd from tracking history on Tensors
# with ``.requires_grad=True`` by wrapping the code block in
# ``with torch.no_grad():``
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
	print((x ** 2).requires_grad)

###############################################################
# **Read Later:**
#
# Document about ``autograd.Function`` is at
# https://pytorch.org/docs/stable/autograd.html#function
```

# 2. 词向量Embedding使用

1. 存储：Embedding存储在$|V|*D$的矩阵中

```python
# Author: Robert Guthrie

# Author: Robert Guthrie

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
word_to_ix = {"hello": 0, "world": 1}
embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
hello_embed = embeds(lookup_tensor)
print(hello_embed)
```

2. 更新权值，使得词向量能代表语料库--n-gram方法

```python
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# We will use Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()
# we should tokenize the input, but we will ignore that for now
# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]
# print the first 3, just so you can see what they look like
print(trigrams[:3])

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    for context, target in trigrams:

        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in tensors)
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(context_idxs)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a tensor)
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()
    losses.append(total_loss)
print(losses)  # The loss decreased every iteration over the training data!
```

3. NLP深度学习中经常使用连续词袋模型（CBOW）。在给定目标词之前和之后的几个词的上下文中，它是一种试图预测词的模型。这与语言建模不同，因为CBOW不是顺序的，并且不一定是概率性的。通常，CBOW用于快速训练词向量 s，而这些向量 s用于初始化某些更复杂模型的向量。通常，这称为*预训练嵌入*。它几乎总是可以将性能提高百分之几。 

词向量的公式是：
$$
-\log p(w_i | C) = -\log \text{Softmax}(A(\sum_{w \in C} q_w) + b)
$$
C是上下文词，w_i是中心词，q_w是w的词向量

