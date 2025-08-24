import heapq
from typing import List, Tuple
from collections import defaultdict
import time
from typing import Optional
from litellm import partial
from minisglang.engine.batch import Req
from minisglang.memory.kvcache import KVCache
from minisglang.memory.page_manager import PageManager
import torch


"""
Radix Tree
key: token ids
value: ppns
"""


class TreeNode:
    counter = 0

    def __init__(self, id: Optional[int] = None):
        self.children = defaultdict(TreeNode)
        self.parent = None
        self.key = None
        self.value = None
        self.lock_ref = 0
        self.last_access_time = time.time()

        self.hit_count = 0
        # indicating the node is loading KV cache from host
        self.loading = False
        # store the host indices of KV cache
        self.host_value = None

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def _key_match_page_size1(key0: List, key1: List):
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i


def _key_match_paged(key0: List, key1: List, page_size: int):
    min_len = min(len(key0), len(key1))

    i = 0
    while i < min_len:
        if key0[i : i + page_size] != key1[i : i + page_size]:
            break
        i += page_size

    return i


class PagedRadixCache:
    def __init__(
        self,
        page_manager: PageManager,
        kvcache: KVCache,
        page_size: Optional[int] = None,
    ):
        self.page_manager = page_manager
        self.kvcache = kvcache
        self.page_size = page_size if page_size is not None else page_manager.page_size
        self.device = kvcache.device if kvcache is not None else torch.device("cpu")

        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = lambda key: key[0]
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=self.page_size)
            self.get_child_key_fn = lambda key: tuple(key[: self.page_size])
        self.reset()

    def reset(self):
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0
        self.protected_size_ = 0

    def match_prefix(self, key: List[int]) -> Tuple[torch.Tensor, TreeNode]:
        """Find the matching prefix from the radix tree.
        Args:
            key: A list of token IDs to find a matching prefix.
        Returns:
            A tuple of a tensor of matching prefix **ppns** and
            the last node that contains the prefix values. Note that
            this API can modify the internal state of the Radix tree.
            The last node create a new child if the prefix is shorter
            than the last node's value.
        """
        if len(key) == 0:
            return (
                torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                self.root_node,
            )

        # align the key
        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)
        return value, last_node

    def insert(self, key: List, value=None):
        """The value should be ppns!!!"""
        assert value is not None
        # Since match_prefix can only match page-aligned keys, we also store page aligned keys in insert here.
        if self.page_size != 1:
            full_pages = len(key) // self.page_size
            key = key[: full_pages * self.page_size]
            value = value[:full_pages]

        return self._insert_helper(self.root_node, key, value)

    def cache_finished_req(self, req: Req):
        token_ids = (req.origin_input_ids + req.output_ids)[:-1]
        full_page_num = len(token_ids) // self.page_size
        ceil_page_num = (len(token_ids) + self.page_size - 1) // self.page_size
        ppns = self.page_manager.page_table[req.page_table_id, :full_page_num].clone()
        free_ppns = self.page_manager.page_table[
            req.page_table_id, full_page_num:ceil_page_num
        ]
        # release the last kv cache that less than one page
        if free_ppns.numel() > 0:
            self.kvcache._free_pages(free_ppns)

        new_prefix_len = self.insert(token_ids[: full_page_num * self.page_size], ppns)
        new_prefix_pages = new_prefix_len // self.page_size
        # 重复部分，从len(req.prefix_ppns)到new_prefix_pages这段的pages是已经存在于kvcache的，可以直接将ppns归还
        self.kvcache._free_pages(ppns[len(req.prefix_ppns) : new_prefix_pages])

        # Remove req slot release the cache lock
        self.page_manager.free([req.page_table_id])
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req):
        token_ids = req.fill_ids
        full_page_num = len(token_ids) // self.page_size
        ppns = self.page_manager.page_table[req.page_table_id, :full_page_num].clone()

        page_aligned_token_ids = token_ids[: full_page_num * self.page_size]

        # 未完成的req之后还需要再forward，故保留其在page_table中的state，这里仅更新其prefix_ppns
        # 归还重复的ppns
        new_prefix_len = self.insert(page_aligned_token_ids, ppns)
        new_prefix_pages = new_prefix_len // self.page_size
        self.kvcache._free_pages(ppns[len(req.prefix_ppns) : new_prefix_pages])

        # 将新加入的前缀写入page_table
        new_prefix_ppns, new_last_node = self.match_prefix(page_aligned_token_ids)
        self.page_manager.write_ppns_decode(
            req.page_table_id,
            torch.arange(len(req.prefix_ppns), len(new_prefix_ppns)),
            new_prefix_ppns[len(req.prefix_ppns) :],
        )

        # 更新引用计数
        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # 更新prefix_ppns和last_node
        req.prefix_ppns = new_prefix_ppns
        req.last_node = new_last_node

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper()

    def evict(self, num_tokens: int):
        leaves: List[TreeNode] = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves) > 0:
            x = heapq.heappop(leaves)

            if x == self.root_node:
                break
            if x.lock_ref > 0:
                continue

            if self.kvcache is not None:
                self.kvcache._free_pages([x.value])
            self._delete_leaf(x)

            num_evicted += len(x.value) * self.page_size
            if len(x.parent.children) == 0:
                heapq.heappush(leaves, x.parent)

    def inc_lock_ref(self, node: TreeNode):
        """Increase the ref cnt of one node (including the whole path towards root)"""
        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.value) * self.page_size
                self.protected_size_ += len(node.value) * self.page_size
                delta -= len(node.value) * self.page_size
            node.lock_ref += 1
            node = node.parent
        return delta

    def dec_lock_ref(self, node: TreeNode):
        """Decrease the ref cnt of one node (including the whole path towards root)"""
        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.value) * self.page_size
                self.protected_size_ -= len(node.value) * self.page_size
                delta += len(node.value) * self.page_size
            node.lock_ref -= 1
            node = node.parent

        return delta

    def evictable_size(self):
        return self.evictable_size_

    def protected_size(self):
        return self.protected_size_

    ##### Internal Helper Functions #####
    def _match_prefix_helper(self, node: TreeNode, key: List) -> Tuple[List, TreeNode]:
        node.last_access_time = time.time()
        child_key = self.get_child_key_fn(key)
        value = []
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.time()
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                # TODO
                pass
            else:
                value.append(child.value)
                node = child
                key = key[prefix_len:]
                if len(key):
                    child_key = self.get_child_key_fn(key)

        return value, node

    def _split_node(self, key, child: TreeNode, split_len: int):
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node
        return new_node

    def _insert_helper(self, node: TreeNode, key: List, value):
        node.last_access_time = time.time()
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.time()
            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len // self.page_size :]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[child_key] = new_node
            self.evictable_size_ += len(value) * self.page_size
        return total_prefix_length

    def _print_helper(self, node: TreeNode, indent: int):
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                len(current_node.key),
                current_node.key[:10],
                f"r={current_node.lock_ref}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == self.get_child_key_fn(
                    child.key
                ), f"{key=}, {self.get_child_key_fn(child.key)=}"

    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(node.key)

    def _total_size_helper(self):
        total_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value) * self.page_size
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size

    def _collect_leaves(self):
        ret_list = []
        stack = [self.root_node]

        while stack:
            cur_node = stack.pop()
            if len(cur_node.children) == 0:
                ret_list.append(cur_node)
            else:
                stack.extend(cur_node.children.values())

        return ret_list


# =========== Testing codes ===========


def _test_insert(tree_cache: PagedRadixCache, key: str):
    tree_cache.insert(
        key,
        torch.arange(0, len(key), device=tree_cache.device, dtype=torch.int64)
    )
    
if __name__ == "__main__":
    tree = PagedRadixCache(None, None, page_size=4)

    _test_insert(tree, "Hello")

    _test_insert(tree, "Hello")

    _test_insert(tree, "Hello_L.A.!")

    _test_insert(tree, "Hello_world! Happy")

    _test_insert(tree, "I love you!")

    tree.pretty_print()

    print(tree.match_prefix("I love you! aha"))

    tree.evict(5)
    tree.pretty_print()
    
    tree.evict(10)
    tree.pretty_print()

