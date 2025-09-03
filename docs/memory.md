# Memory

我们将SGlang的`ReqToToken`和`TokenToKVPool`的两级内存结构改为`PageManager`和`KVCache`。这和原来的设计有一点微小的不同。在`PageManager`中，我们直接存储了一个“页表”，而不是存下每个token在KVCache中的地址。这样在一定程度上减小了地址的存储冗余和`PageManager`的大小。`KVCache`就是实际的KV cache管理单元。

`PageManager`的关键成员`page_table`是一个形状为`(self.max_req_num, self.max_page_num)`的`Tensor`，提供形如
```
(req_id, vpn) -> ppn
```
的索引。这里的vpn就是一个页在一个Req中的页号，ppn就是这个页在实际的`KVCache`中的页号。我们所有的索引操作都按照页对齐，尽量使用页号而不是每个token的地址。这也使FlashAttention的调用更简便。

按页对齐的含义是：

对于某个Req，它的第$m \cdot \text{page\_size} + n$个token在`KVCache`中的地址总是类似$p \cdot \text{page\_size} + q$，其中$m,n,p,q$都是整数，且$n, q\in [0, \text{page\_size})$。这一点是我们人为保证的。