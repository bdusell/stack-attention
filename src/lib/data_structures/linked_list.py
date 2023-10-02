class LinkedList:

    def __init__(self, values):
        self.head = None
        self.tail = None
        for value in values:
            self.append(value)

    def append(self, value):
        self.append_node(LinkedListNode(value))

    def append_node(self, node):
        if self.head is None:
            self.head = self.tail = node
        else:
            self.tail.next = node
            self.tail = node

    def extend(self, linked_list):
        # Constant time. Does not modify the argument linked_list.
        if self.head is None:
            self.head = linked_list.head
        else:
            self.tail.next = linked_list.head
        self.tail = linked_list.tail

    def values(self):
        curr = self.head
        while curr is not None:
            yield curr.value
            curr = curr.next

    def __iter__(self):
        return iter(self.values())

class LinkedListNode:

    def __init__(self, value):
        self.value = value
        self.next = None
